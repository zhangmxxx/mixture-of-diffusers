import argparse
import json
from pathlib import Path
from PIL import Image
from diffusers import LMSDiscreteScheduler, DDIMScheduler # type: ignore
from diffusers import UNet2DConditionModel # type: ignore
from mixdiff import StableDiffusionCanvasPipeline, Text2ImageRegion, Image2ImageRegion, preprocess_image
import torch

def generate_grid(generation_arguments):
  model_id = "CompVis/stable-diffusion-v1-4"
  # Prepared scheduler
  if generation_arguments["scheduler"] == "ddim":
    scheduler = DDIMScheduler()
  elif generation_arguments["scheduler"] == "lms":
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
  else:
    raise ValueError(f"Unrecognized scheduler {generation_arguments['scheduler']}")
  # unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16)
  # can deduce dtype for unet, vae...
  pipe = StableDiffusionCanvasPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:5")
  # add more args if needed
  pipeargs = {
    "num_inference_steps": generation_arguments["steps"],
    "seed": generation_arguments["seed"],
    "regions": generation_arguments["regions"],
    "canvas_height": generation_arguments["canvas_height"], 
    "canvas_width": generation_arguments["canvas_width"], 
    "cpu_vae": generation_arguments["cpu_vae"] if "cpu_vae" in generation_arguments else False,
  }
  # Mixture of Diffusers generation
  image = pipe(**pipeargs)["sample"][0]
  outname = "output"
  outpath = "./outputs"
  Path(outpath).mkdir(parents=True, exist_ok=True)
  image.save(f"{outpath}/{outname}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a stable diffusion grid using a JSON file with all configuration parameters.')
    parser.add_argument('config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    with open(args.config, "r") as f:
      generation_arguments = json.load(f)
    # parse prompt and generate Text2Image / Imgage2Imgage list
    saved_list = []
    for item in generation_arguments["regions"]:
      if "ref_img" in item:
        iic_image = preprocess_image(Image.open(item["ref_img"]).convert("RGB"))
        saved_list.append(Image2ImageRegion(
          row_init = item["area"][0],
          row_end  = item["area"][1],
          col_init = item["area"][2],
          col_end  = item["area"][3],
          reference_image = iic_image, # type: ignore
          strength= item["strength"]
        ))
      else:
        saved_list.append(Text2ImageRegion(
          row_init = item["area"][0],
          row_end  = item["area"][1],
          col_init = item["area"][2],
          col_end  = item["area"][3],
          guidance_scale = item["guidance_scale"],
          prompt = item["prompt"]
        ))
    generation_arguments["regions"] = saved_list
    generate_grid(generation_arguments)