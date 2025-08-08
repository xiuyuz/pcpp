import torch
from packaging import version
from torch import distributed as dist
from torch.distributed import P2POp
import os

def check_env():
    if version.parse(torch.version.cuda) < version.parse("11.3"):
        # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
        raise RuntimeError("NCCL CUDA Graph support requires CUDA 11.3 or above")
    if version.parse(version.parse(torch.__version__).base_version) < version.parse("2.2.0"):
        # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
        raise RuntimeError(
            "CUDAGraph with NCCL support requires PyTorch 2.2.0 or above. "
            "If it is not released yet, please install nightly built PyTorch with "
            "`pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121`"
        )


def is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


class DistriConfig:
    def __init__(
        self,
        height: int = 1024,
        width: int = 1024,
        do_classifier_free_guidance: bool = True,
        split_batch: bool = True,
        warmup_steps: int = 4,
        comm_checkpoint: int = 20, # original=60
        mode: str = "corrected_async_gn", # 
        use_cuda_graph: bool = True,
        parallelism: str = "patch", # patch
        split_scheme: str = "row",
        verbose: bool = False,
    ):
        try:
            # Initialize the process group
            # os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
            dist.init_process_group("nccl")
            # Get the rank and world_size
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except Exception as e:
            rank = 0
            world_size = 1
            print(f"Failed to initialize process group: {e}, falling back to single GPU")

        assert is_power_of_2(world_size)
        check_env()

        self.world_size = world_size
        self.rank = rank
        self.height = height
        self.width = width
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.split_batch = split_batch
        self.warmup_steps = warmup_steps
        self.comm_checkpoint = comm_checkpoint
        self.mode = mode
        self.use_cuda_graph = use_cuda_graph

        self.parallelism = parallelism
        self.split_scheme = split_scheme

        self.verbose = verbose

        if do_classifier_free_guidance and split_batch:
            n_device_per_batch = world_size // 2
            if n_device_per_batch == 0:
                n_device_per_batch = 1
        else:
            n_device_per_batch = world_size

        self.n_device_per_batch = n_device_per_batch

        self.height = height
        self.width = width
        local_rank = int(os.environ['LOCAL_RANK']) 
        device = torch.device(f"cuda:{local_rank}") # for multinode, each node has 4 GPUs
        torch.cuda.set_device(device)
        self.device = device

        batch_group = None
        split_group = None
        if do_classifier_free_guidance and split_batch and world_size >= 2:
            batch_groups = []
            for i in range(2):
                batch_groups.append(dist.new_group(list(range(i * (world_size // 2), (i + 1) * (world_size // 2)))))
            batch_group = batch_groups[self.batch_idx()]
            split_groups = []
            for i in range(world_size // 2):
                split_groups.append(dist.new_group([i, i + world_size // 2]))
            split_group = split_groups[self.split_idx()]
        self.batch_group = batch_group
        self.split_group = split_group

    def batch_idx(self, rank: int or None = None) -> int:
        if rank is None:
            rank = self.rank
        if self.do_classifier_free_guidance and self.split_batch:
            return 1 - int(rank < (self.world_size // 2))
        else:
            return 0  # raise NotImplementedError

    def split_idx(self, rank: int or None = None) -> int:
        if rank is None:
            rank = self.rank
        return rank % self.n_device_per_batch
    
    # Return the ranks of the neighbors for PCPP
    def get_neighbor_ranks(self):
        # Returns the ranks of the neighbors
        if self.do_classifier_free_guidance and self.split_batch:
            # upper and lower neighbor
            # [0,1,2,3] [4,5,6,7] 
            # if rank=2, return [1,3] if rank=3, return [2] if rank=4, return [5]
            neighbors = [None, None]
            rank = self.rank % self.n_device_per_batch
            world_size = self.world_size // 2
            if rank > 0:
                neighbors[0] = rank - 1  # left neighbor
            if rank < world_size - 1:
                neighbors[1] = rank + 1  # right neighbor
            if self.rank >= self.n_device_per_batch:  # self.rank not rank!!!
                for i in range(2):
                    if neighbors[i] is not None:
                        neighbors[i] += self.n_device_per_batch
            return neighbors
        else:
            neighbors = [None, None]
            rank = self.rank
            world_size = self.world_size
            if rank > 0:
                neighbors[0] = rank - 1
            if rank < world_size - 1:
                neighbors[1] = rank + 1
            return neighbors


class PatchParallelismCommManager:
    def __init__(self, distri_config: DistriConfig):
        self.distri_config = distri_config

        self.torch_dtype = None
        self.numel = 0
        self.numel_dict = {}

        self.buffer_list = None

        self.starts = []
        self.ends = []
        self.shapes = []

        self.idx_queue = []

        self.handles = None
        # info
        self.layer_list = []

    def register_tensor(
        self, shape: tuple[int, ...] or list[int], torch_dtype: torch.dtype, layer_type: str = None
    ) -> int:
        if self.torch_dtype is None:
            self.torch_dtype = torch_dtype
        else:
            assert self.torch_dtype == torch_dtype
        self.starts.append(self.numel)
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel += numel
        if layer_type is not None:
            if layer_type not in self.numel_dict:
                self.numel_dict[layer_type] = 0
            self.numel_dict[layer_type] += numel

        self.ends.append(self.numel)
        self.shapes.append(shape)
        self.layer_list.append(layer_type)
        return len(self.starts) - 1

    def create_buffer(self):
        distri_config = self.distri_config
        if distri_config.rank == 0:
            print(
                f"Create buffer with {self.numel / 1e6:.3f}M parameters for {len(self.starts)} tensors on each device."
            )
            for layer_type, numel in self.numel_dict.items():
                print(f"  {layer_type}: {numel / 1e6:.3f}M parameters")
            print(f'layer_list: {self.layer_list}')

        self.buffer_list = [
            torch.empty(self.numel, dtype=self.torch_dtype, device=self.distri_config.device)
            for _ in range(self.distri_config.n_device_per_batch)
        ]
        self.handles = [None for _ in range(len(self.starts))]


    def get_buffer_list(self, idx: int) -> list[torch.Tensor]:
        buffer_list = [t[self.starts[idx] : self.ends[idx]].view(self.shapes[idx]) for t in self.buffer_list]
        return buffer_list

    def communicate(self):
        distri_config = self.distri_config
        start = self.starts[self.idx_queue[0]]
        end = self.ends[self.idx_queue[-1]]
        tensor = self.buffer_list[distri_config.split_idx()][start:end]
        buffer_list = [t[start:end] for t in self.buffer_list]
        handle = dist.all_gather(buffer_list, tensor, group=self.distri_config.batch_group, async_op=True)
        for i in self.idx_queue:
            self.handles[i] = handle
        self.idx_queue = []

    def enqueue(self, idx: int, tensor: torch.Tensor):
        distri_config = self.distri_config
        # complete one forward pass, back to first layer.
        if idx == 0 and len(self.idx_queue) > 0:
            self.communicate()
        assert len(self.idx_queue) == 0 or self.idx_queue[-1] == idx - 1
        # insert the new value to the device idx and the buffer.
        self.idx_queue.append(idx)
        self.buffer_list[distri_config.split_idx()][self.starts[idx] : self.ends[idx]].copy_(tensor.flatten())
        # comm_checkpoint = 60. how many layers?
        if len(self.idx_queue) == distri_config.comm_checkpoint:
            self.communicate()

    def clear(self):
        if len(self.idx_queue) > 0:
            self.communicate()
        if self.handles is not None:
            for i in range(len(self.handles)):
                if self.handles[i] is not None:
                    self.handles[i].wait()
                    self.handles[i] = None


# Async send and receive per layer
class PCPPCommManager(PatchParallelismCommManager):
    def __init__(self, distri_config):
        super().__init__(distri_config)
        # Initializes neighbors based on the distributed configuration
        self.PCPP_idx = 0
        self.neighbors = self.distri_config.get_neighbor_ranks()

        print(f'rank: {self.distri_config.rank}, neighbors: {self.neighbors}')
        self.requests = None
        self.PCPP_recv_buffers = None

    def get_PCPP_idx(self):
        idx = self.PCPP_idx
        self.PCPP_idx += 1
        return idx
    
    # call once after the first forward pass
    def create_buffer(self):
        super().create_buffer()
        self.PCPP_recv_buffers = [None for _ in range(self.PCPP_idx)]
        distri_config = self.distri_config
        if distri_config.rank == 0 and distri_config.verbose:
            print(
                f"Create PCPP Buffers with {self.PCPP_idx} layers in total."
            )


    def send_to_neighbors_async(self, idx, send_tensor):
        # Clear previous communication requests if any
        self.wait_all_requests()

        # Setup batched communication operations
        p2p_op_list = []
        self.PCPP_recv_buffers[idx] = [None for _ in range(2)]
        
        for i, neighbor in enumerate(self.neighbors):
            if neighbor is None:
                continue
            recv_buffer = torch.empty_like(send_tensor, device=self.distri_config.device)
            self.PCPP_recv_buffers[idx][i] = recv_buffer

            # Prepare the send and receive operations
            send_op = P2POp(dist.isend, send_tensor, neighbor)
            recv_op = P2POp(dist.irecv, recv_buffer, neighbor)

            p2p_op_list.append(send_op)
            p2p_op_list.append(recv_op)

        # Execute the batched send and receive operations
        
        self.requests = dist.batch_isend_irecv(p2p_op_list)
        if self.distri_config.rank == 0 and self.distri_config.verbose:
            print(f'idx: {idx}, rank: {self.distri_config.rank}')
            print(f'p2p_op_list: {p2p_op_list}')
            print(f'Sent ASYNC requests to neighbors: {self.neighbors}')
            print(f'requests: {self.requests}')
            
        # handle the last layer's communication so that it can start 
        # a new forward fresh without any pending requests
        # if idx == self.PCPP_idx - 1:
        #     print(f'Handling last layer communication')
        #     self.wait_all_requests()
        #     print(f'recv buffer at last idx: {self.PCPP_recv_buffers[idx]}')
        #     print(f'PP index queue: {self.idx_queue}')
        #     self.communicate()
        #     print(f'PP index queue after communicate: {self.idx_queue}')
            
            
    def wait_all_requests(self):
        # Wait for all ongoing send and receive operations to complete
        if self.requests:
            if self.distri_config.rank == 0 and self.distri_config.verbose:
                print(f'Waiting for requests to complete: {self.requests}')
            for req in self.requests:
                req.wait()
                if self.distri_config.rank == 0 and self.distri_config.verbose:
                    print(f'Completed request: {req}')
        self.requests = []


    def clear(self):
        # Ensure all communication operations are completed before clearing
        self.wait_all_requests()
        super().clear()  # Clear handles and potential remaining tasks
