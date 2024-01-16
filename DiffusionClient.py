from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import pt_to_pil, load_image
import torch
import discord
from discord.ext import commands 
import io
import PIL
import math
import concurrent.futures
import asyncio
import gc

class ImageGenerator(commands.Cog):
    def __init__(self, bot, chat_cog):
        self.bot = bot
        self.pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipeline.to("cuda:1")

        self.chat_cog = chat_cog
        self.lock = asyncio.Lock()

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
        image = self.pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        return image

    async def generate_image_async(self, ctx, prompt):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            image = await loop.run_in_executor(pool, self.generate_image_blocking, prompt)
        return image

    @commands.command(aliases=["paint"])
    async def generate_image(self, ctx, *, prompt):
        """Generate an image based on the given prompt."""
        async with self.lock:
            async with ctx.typing():
                image = await self.generate_image_async(ctx, prompt)
                torch.cuda.empty_cache()
                gc.collect()

                if image:
                    with io.BytesIO() as binary_img:
                        image.save(binary_img, 'PNG')
                        binary_img.seek(0)
                        file = discord.File(binary_img, filename='image.png')
                        await ctx.send(file=file)
                else:
                    await ctx.send("Unable to generate an image for the given prompt.")

    @commands.command(name="pipe")
    async def pipe(self, ctx, *, prompt):
        result = await self.chat_cog.piped_chat(ctx, prompt)
        result = await self.generate_image_async(ctx, result)
        torch.cuda.empty_cache()
        gc.collect()

        if isinstance(result, PIL.Image.Image):
            with io.BytesIO() as binary_img:
                result.save(binary_img, 'PNG')
                binary_img.seek(0)
                file = discord.File(binary_img, filename='result.png')
                await ctx.send(file=file)
        else:
            await ctx.send("Invalid piped command.")


async def setup_diffusion_client(bot):
    chat_cog = bot.get_cog("ChatGenerator")
    #chat_cog = True
    if chat_cog and not bot.get_cog("ImageGenerator"):
        #chat_cog = None
        await bot.add_cog(ImageGenerator(bot, chat_cog))
    else:
        print("ImageGenerator cog has already been added.")