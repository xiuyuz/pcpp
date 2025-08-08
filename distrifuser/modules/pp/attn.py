import torch
import math
from diffusers.models.attention import Attention
from diffusers.utils import USE_PEFT_BACKEND
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from distrifuser.modules.base_module import BaseModule
from distrifuser.utils import DistriConfig


class DistriAttentionPP(BaseModule):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriAttentionPP, self).__init__(module, distri_config)

        to_k = module.to_k
        to_v = module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape

        in_size, out_size = to_k.in_features, to_k.out_features
        to_kv = nn.Linear(
            in_size,
            out_size * 2,
            bias=to_k.bias is not None,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        to_kv.weight.data[:out_size].copy_(to_k.weight.data)
        to_kv.weight.data[out_size:].copy_(to_v.weight.data)

        if to_k.bias is not None:
            assert to_v.bias is not None
            to_kv.bias.data[:out_size].copy_(to_k.bias.data)
            to_kv.bias.data[out_size:].copy_(to_v.bias.data)

        self.to_kv = to_kv


class DistriCrossAttentionPP(DistriAttentionPP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriCrossAttentionPP, self).__init__(module, distri_config)
        self.kv_cache = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor or None = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        assert encoder_hidden_states is not None
        recompute_kv = self.counter == 0

        attn = self.module
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if recompute_kv or self.kv_cache is None:
            kv = self.to_kv(encoder_hidden_states)
            self.kv_cache = kv
        else:
            kv = self.kv_cache
        key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.counter += 1

        return hidden_states


class DistriSelfAttentionPP(DistriAttentionPP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriSelfAttentionPP, self).__init__(module, distri_config)

    def _forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0):
        attn = self.module
        distri_config = self.distri_config
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        encoder_hidden_states = hidden_states

        kv = self.to_kv(encoder_hidden_states)

        if distri_config.n_device_per_batch == 1:
            full_kv = kv
        else:
            # buffer_list still contains activations from all devices, even if we are not using them
            # in the future PCPP steps.
            if self.buffer_list is None:  # buffer not created
                # if dist.get_rank() == 0:
                #     print(f"First forward pass full kv")
                full_kv = torch.cat([kv for _ in range(distri_config.n_device_per_batch)], dim=1)
            elif distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps:
                # if dist.get_rank() == 0:
                #     print(f"performing all_gather for layer {self.idx}")
                dist.all_gather(self.buffer_list, kv, group=distri_config.batch_group, async_op=False)
                full_kv = torch.cat(self.buffer_list, dim=1)
            else:
                # n activations from warmup steps
                new_buffer_list = [buffer for buffer in self.buffer_list]

                # insert the new activation (curr device) into its position
                new_buffer_list[distri_config.split_idx()] = kv

                # insert new activations from neighbors
                if self.comm_manager and self.comm_manager.PCPP_recv_buffers is not None:
                    if self.comm_manager.PCPP_recv_buffers[self.idx]:
                        if dist.get_rank() == 0 and self.distri_config.verbose:
                            print(f"inserting new activations from neighbors for layer {self.idx}")
                        for i, buffer in enumerate(self.comm_manager.PCPP_recv_buffers[self.idx]):
                            # upper neighbor
                            if i == 0 and buffer is not None:
                                new_buffer_list[distri_config.split_idx() - 1] = buffer
                            # lower neighbor
                            if i == 1 and buffer is not None:
                                new_buffer_list[distri_config.split_idx() + 1] = buffer

                full_kv = torch.cat(new_buffer_list, dim=1)
                # async all_gather the new activation kv to other devices
                if distri_config.mode != "no_sync":
                    # print(f'rank {dist.get_rank()} enqueueing buffer for layer {self.idx}')
                    self.comm_manager.send_to_neighbors_async(self.idx, kv)
                    # if dist.get_rank() == 0 and self.idx > 0:
                        # print(f'Received buffer from neighbors for previous idx {self.idx-1}')
                        # print(f'PCPP_recv_buffers: {self.comm_manager.PCPP_recv_buffers[self.idx-1][1]}')
                
                # ---------------------------------------------------------------------------- #
                #                               Edit partial here                              #
                # ---------------------------------------------------------------------------- #
                portion = 0.3
                # ---------------------------------------------------------------------------- #
                #                               Edit partial here                              #
                # ---------------------------------------------------------------------------- #
                current_idx = distri_config.split_idx()
                kv_length = kv.shape[1]
                start_idx = int(max(0, kv_length * (current_idx - 1 * portion)))
                end_idx = int(min(kv_length * (portion * 1 + 1 + current_idx), full_kv.shape[1]))
                full_kv = full_kv[:, start_idx:end_idx, :]
            
        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)
        # key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor or None = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        distri_config = self.distri_config
        # if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
        #     if self.comm_manager.handles[self.idx] is not None:
        #         self.comm_manager.handles[self.idx].wait()
        #         self.comm_manager.handles[self.idx] = None

        # PCPP - receive the kv from neighbors if needed
        # if self.comm_manager is not None:
        #     # if in the first forward pass, receive from neighbors
        #     self.comm_manager.wait_all_requests()
            

        b, l, c = hidden_states.shape
        if distri_config.n_device_per_batch > 1 and self.buffer_list is None:
            # if still in the first forward pass, register for this one time.
            # since buffer_list is not created until the end of the first pass
            # this works for the PCPP case as well.
            if self.comm_manager.buffer_list is None:
                self.idx = self.comm_manager.get_PCPP_idx()
            else:
                self.buffer_list = [torch.empty((b, l, self.to_kv.out_features), 
                                                dtype=hidden_states.dtype, 
                                                device=distri_config.device) 
                                    for _ in range(distri_config.n_device_per_batch)]
                
        output = self._forward(hidden_states, scale=scale)

        self.counter += 1
        return output
