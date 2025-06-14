from diffusers import StableDiffusionPipeline
import torch

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model without revision
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Generate an image
prompt = "a fantasy castle floating in the sky"
image = pipe(prompt).images[0]

# Save the image
image.save("fantasy_castle.png")
