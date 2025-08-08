# PCPP: Partially Conditioned Patch Parallelism for Accelerated Diffusion Model Inference

### [Paper](https://arxiv.org/abs/2412.02962)

![teaser](assets/pcpp.png)
*Diffusion models have exhibited exciting capabilities in generating images and are also very promising for video creation. However, the inference speed of diffusion models is limited by the slow sampling process, restricting its use cases. The sequential denoising steps required for generating a single sample could take tens or hundreds of iterations and thus have become a significant bottleneck. This limitation is more salient for applications that are interactive in nature or require small latency. To address this challenge, we propose Partially Conditioned Patch Parallelism (PCPP) to accelerate the inference of high-resolution diffusion models. Using the fact that the difference between the images in adjacent diffusion steps is nearly zero, Patch Parallelism (PP) leverages multiple GPUs communicating asynchronously to compute patches of an image in multiple computing devices based on the entire image (all patches) in the previous diffusion step. PCPP develops PP to reduce computation in inference by conditioning only on parts of the neighboring patches in each diffusion step, which also decreases communication among computing devices. As a result, PCPP decreases the communication cost by around 70% compared to DistriFusion (the state of the art implementation of PP) and achieves 2.36 ~ 8.02X inference speed-up using 4 ~ 8 GPUs compared to 2.32 ~ 6.71X achieved by DistriFusion depending on the computing device configuration and resolution of generation at the cost of a possible decrease in image quality. PCPP demonstrates the potential to strike a favorable trade-off, enabling high-quality image generation with substantially reduced latency.*

Partially Conditioned Patch Parallelism for Accelerated Diffusion Model Inference
</br>
XiuYu Zhang, Zening Luo, Michelle E. Lu</br>
UC Berkeley</br>
2024

## Overview
![idea](https://arxiv.org/html/2412.02962v1/x2.png)
Building on [DistriFusion](https://github.com/mit-han-lab/distrifuser?tab=readme-ov-file), we introduce our PCPP for parallelizing the inference of diffusion models using multiple devices asynchronously by point-to-point communication. The key idea is to partition the image horizontally into non-overlapping patches and process each patch conditioned on only itself and the parts of its neighboring patches (from the previous step) on separate devices. This approach is based on the hypothesis that generating image patches does not always necessitate dependency on all other patches; instead, satisfactory results can be achieved by relying solely on neighboring patches (for some cases). Please refer to the paper for details.

The major difference in communication is shown as the replacement of `PatchParallelismCommManager` class in DistriFusion to our `PCPPCommManager` class defined in `distrifuser/utils.py`. We also made necessary changes in `distrifuser/models/base_model.py`, `distrifuser/models/distri_sdxl_unet_pp.py`, `distrifuser/modules/pp/attn.py`, and `distrifuser/pipelines.py`.



## Prerequisites

* Python3
* NVIDIA GPU + CUDA >= 12.0 and corresponding CuDNN
* [PyTorch](https://pytorch.org) >= 2.2.


## Usage Example (adapted from DistriFusion)

In `scripts/sdxl_example.py`, we provide a minimal script for running [SDXL](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl) with PCPP. 

```python
import torch
import json

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, split_batch=True)
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

    
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
    num_inference_steps = 20,
).images[0]
if distri_config.rank == 0:
    image.save("doctor-PCPP-0.3.png")
```
The running command is 
```shell
torchrun --nproc_per_node=$N_GPUS scripts/sdxl_example.py
```

where `$N_GPUS` is the number GPUs you want to use.

Specifically, our `distrifuser` shares the same APIs as DistriFusion and can be used in a similar way.  The default partial value for PCPP generation is 0.3, and as for now, it needs to be changed manually in `distrifuser/modules/pp/attn.py`.
```python

    # ---------------------------------------------------------------------------- #
    #                               Edit partial here                              #
    # ---------------------------------------------------------------------------- #
    portion = 0.3
    # ---------------------------------------------------------------------------- #
    #                               Edit partial here                              #
    # ---------------------------------------------------------------------------- #

```


### Benchmark (adapted from DistriFusion)

Our benchmark results are using [PyTorch](https://pytorch.org) 2.2 and [diffusers](https://github.com/huggingface/diffusers) 0.24.0. First, you may need to install some additional dependencies:

```shell
pip install git+https://github.com/zhijian-liu/torchprofile datasets torchmetrics dominate clean-fid
```

#### COCO Quality

You can use `scripts/generate_coco.py` to generate images with COCO captions. The command is

```
torchrun --nproc_per_node=$N_GPUS scripts/generate_coco.py --no_split_batch
```

where `$N_GPUS` is the number GPUs you want to use. By default, the generated results will be stored in `results/coco`. You can also customize it with `--output_root`. Some additional arguments that you may want to tune:

* `--num_inference_steps`: The number of inference steps. We use 50 by default.
* `--guidance_scale`: The classifier-free guidance scale. We use 5 by default.
* `--scheduler`: The diffusion sampler. We use [DDIM sampler](https://huggingface.co/docs/diffusers/v0.26.3/en/api/schedulers/ddim#ddimscheduler) by default.
* `--warmup_steps`: The number of additional warmup steps (4 by default). 
* `--sync_mode`: Different GroupNorm synchronization modes. By default, it is using our corrected asynchronous GroupNorm.
* `--parallelism`: The parallelism paradigm you use. By default, it is patch parallelism implemented following PCPP. You can use `tensor` for tensor parallelism and `naive_patch` for naïve patch.

After you generate all the images, you can use our script `scripts/compute_metrics.py` to calculate PSNR, LPIPS and FID. The usage is 

```shell
python scripts/compute_metrics.py --input_root0 $IMAGE_ROOT0 --input_root1 $IMAGE_ROOT1
```

where `$IMAGE_ROOT0` and `$IMAGE_ROOT1` are paths to the image folders you are trying to comparing. If `IMAGE_ROOT0` is the ground-truth foler, please add a `--is_gt` flag for resizing. We also provide a script `scripts/dump_coco.py` to dump the ground-truth images.


## Citation

If you find our work useful, please consider citing it in your own research.

```bibtex
@misc{zhang2024pcpp,
      title={Partially Conditioned Patch Parallelism for Accelerated Diffusion Model Inference}, 
      author={XiuYu Zhang and Zening Luo and Michelle E. Lu},
      year={2024},
      eprint={2412.02962},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.02962}, 
}
```

## Acknowledgments

Our code is developed based on [DistriFusion](https://github.com/mit-han-lab/distrifuser) and thus adapted the same MIT license.
