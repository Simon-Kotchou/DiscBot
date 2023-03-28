from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import discord
from discord.ext import commands 
import io

class ImageGenerator(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.repo_id = "stabilityai/stable-diffusion-2"
        self.pipe = DiffusionPipeline.from_pretrained(self.repo_id, torch_dtype=torch.float16, revision="fp16")

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

    @commands.command(aliases=["paint"])
    async def generate_image(self, ctx, *, prompt):
        """Generate an image based on the given prompt."""
        async with ctx.typing():
            with torch.no_grad():
                image = self.pipe(prompt, guidance_scale=9, num_inference_steps=100).images[0]

            if image:
                with io.BytesIO() as binary_img:
                    image.save(binary_img, 'PNG')
                    binary_img.seek(0)
                    file = discord.File(binary_img, filename='image.png')
                    await ctx.send(file=file)

            else:
                await ctx.send("Unable to generate an image for the given prompt.")

async def setup_diffusion_client(bot):
    if not bot.get_cog("Diffusion"):
        await bot.add_cog(ImageGenerator(bot))
    else:
        print("Music cog has already been added.")