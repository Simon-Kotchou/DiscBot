from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInstructPix2PixPipeline
import torch
import discord
from discord.ext import commands 
import io
import PIL
import math
import concurrent.futures
import asyncio

class ImageGenerator(commands.Cog):
    def __init__(self, bot):
        torch.cuda.empty_cache()
        self.bot = bot
        self.device_1 = "cuda:0"
        self.device_2 = "cuda:1"
        self.repo_id_gen = "stabilityai/stable-diffusion-2"
        self.pipe_gen = DiffusionPipeline.from_pretrained(self.repo_id_gen, torch_dtype=torch.float16, revision="fp16")

        self.pipe_gen.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe_gen.scheduler.config)
        self.pipe_gen = self.pipe_gen.to(self.device_1)

        self.repo_id_edit = "timbrooks/instruct-pix2pix"
        self.pipe_edit = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.repo_id_edit, torch_dtype=torch.float16).to(self.device_2)

    def process_image(self, image_bytes):
        input_image = PIL.Image.open(io.BytesIO(image_bytes))
        input_image = PIL.ImageOps.exif_transpose(input_image)
        input_image = input_image.convert("RGB")

        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = PIL.ImageOps.fit(input_image, (width, height), method=PIL.Image.Resampling.LANCZOS)
        return input_image

    def generate_image_blocking(self, prompt):
        with torch.no_grad():
            with torch.cuda.device(0):
                image = self.pipe_gen(prompt, guidance_scale=9, num_inference_steps=100).images[0]
        return image

    async def generate_image_async(self, ctx, prompt):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            image = await loop.run_in_executor(pool, self.generate_image_blocking, prompt)
        return image

    @commands.command(aliases=["paint"])
    async def generate_image(self, ctx, *, prompt):
        """Generate an image based on the given prompt."""
        async with ctx.typing():
            image = await self.generate_image_async(ctx, prompt)

            if image:
                with io.BytesIO() as binary_img:
                    image.save(binary_img, 'PNG')
                    binary_img.seek(0)
                    file = discord.File(binary_img, filename='image.png')
                    await ctx.send(file=file)
            else:
                await ctx.send("Unable to generate an image for the given prompt.")
    
    def edit_image_blocking(self, prompt, image):
        with torch.no_grad():
            with torch.cuda.device(1):
                image = self.pipe_edit(prompt, image=image, num_inference_steps=300, image_guidance_scale=1.5, guidance_scale=7).images[0]
        return image

    async def edit_image_async(self, ctx, prompt, image):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            image = await loop.run_in_executor(pool, self.edit_image_blocking, prompt, image)
        return image

    @commands.command(aliases=["edit"])
    async def edit_image(self, ctx, *, prompt):
        # Check if there's an attachment (image) in the message
        if not ctx.message.attachments:
            await ctx.send("Please provide an image attachment.")
            return

        attachment = ctx.message.attachments[0]
        image_bytes = await attachment.read()
        image = self.process_image(image_bytes)

        async with ctx.typing():
            image = await self.edit_image_async(ctx, prompt, image)

            if image:
                with io.BytesIO() as binary_img:
                    image.save(binary_img, 'PNG')
                    binary_img.seek(0)
                    file = discord.File(binary_img, filename='edit.png')
                    await ctx.send(file=file)
            else:
                await ctx.send("Unable to generate an image for the given prompt.")

async def setup_diffusion_client(bot):
    if not bot.get_cog("ImageGenerator"):
        await bot.add_cog(ImageGenerator(bot))
    else:
        print("Music cog has already been added.")