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

# Dumps the pipeline configuration to a JSON file
# with open("pipeline_config.json", "w") as f:
#     json.dump(pipeline.pipeline.components, f, indent=2)
# print(pipeline.pipeline.components)
    
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
    num_inference_steps = 20,
).images[0]
if distri_config.rank == 0:
    image.save("doctor-PCPP-0.3.png")
