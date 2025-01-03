import os
import json
import torch
from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline
from diffusers import StableDiffusion3Pipeline

# Load the JSON file
with open('/home/ubuntu/datasets/FRONT/relationships_anyscene.json', 'r') as file:
    data = json.load(file)

# Loop through each scene in the JSON
for i, scene in enumerate(data['scans']):
    print(scene['scan'], scene['bgprompts'], )
    scan_name = scene['scan']
    bg_prompts = scene['bgprompts']
    source = scene['source']
    style = scene['bgstyle']
    
    # Create the directory for the scene
    scene_dir = f"data/msnd/{scan_name}/"
    os.makedirs(scene_dir, exist_ok=True)
    
    # Load Stable Diffusion pipeline
    sd_pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large")
    
    prompt = bg_prompts['positive'] + f"from {source}. Style: {style}"
    # Generate images for the range of seeds
    for seed in range(2025, 2030):
        # Generate the image
        generator = torch.manual_seed(seed)
        image = sd_pipeline(
            prompt=prompt,
            negative_prompt=bg_prompts['negative'],
            generator=generator,
        ).images[0]
        
        # Save the generated image
        image_path = os.path.join(scene_dir, f"{seed:04d}_color.png")
        image.save(image_path)
        
        # Configure the pipeline for the next step
        cfg = load_cfg('pipe/cfgs/basic.yaml')
        cfg.scene.input.rgb = image_path
        
        # Run the custom pipeline
        vistadream = Pipeline(cfg)
        vistadream()
