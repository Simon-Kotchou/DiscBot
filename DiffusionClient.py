from diffusers import AutoPipelineForText2Image, StableCascadePriorPipeline, StableCascadeDecoderPipeline
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
        self.pipeline = None  # Start without a loaded model
        self.device = "cuda:1"  # Specify the GPU device
        self.device2 = "cuda:0"
        self.chat_cog = chat_cog
        self.lock = asyncio.Lock()
        self.model_quality = None  # Keep track of the currently loaded model quality

    async def load_model(self, quality):
        if self.pipeline is not None:
            del self.pipeline
            torch.cuda.empty_cache()
            gc.collect()

        if quality == "fast":
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", 
                torch_dtype=torch.float16, 
                variant="fp16"
            )
            self.pipeline.to(self.device)
        elif quality == "quality":
            self.pipeline = {
                "prior": StableCascadePriorPipeline.from_pretrained(
                    "stabilityai/stable-cascade-prior", 
                    torch_dtype=torch.bfloat16
                ).to(self.device),
                "decoder": StableCascadeDecoderPipeline.from_pretrained(
                    "stabilityai/stable-cascade",  
                    torch_dtype=torch.float16,
                    revision="refs/pr/17"
                ).to(self.device2)
            }
        else:
            raise ValueError("Invalid quality setting. Choose 'fast' or 'quality'.")

        self.model_quality = quality

    @commands.command(name="load")
    async def load(self, ctx, quality: str):
        """Load a model based on the specified quality ('fast' or 'quality')."""
        async with self.lock:
            try:
                await self.load_model(quality)
                await ctx.send(f"Model loaded with {quality} settings.")
            except ValueError as e:
                await ctx.send(str(e))

    def generate_with_sdxl_turbo(self, prompt):
        image = self.pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, num_images_per_prompt=4).images
        return image

    def generate_with_stable_cascade(self, prompt):
        # Generate embeddings with the "prior" on cuda:1
        prior_output = self.pipeline["prior"](
            prompt=prompt,
            height=1024,
            width=1024,
            negative_prompt="",
            guidance_scale=4.0,
            num_inference_steps=30,
            num_images_per_prompt=1
        )

        # Move the embeddings to cuda:0 before passing to the "decoder"
        embeddings_on_cuda_0 = prior_output.image_embeddings.half().to("cuda:0")

        # Generate the final image with the "decoder" on cuda:0
        decoder_output = self.pipeline["decoder"](
            image_embeddings=embeddings_on_cuda_0,
            prompt=prompt,
            negative_prompt="",
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=20,
        ).images

        return decoder_output

    @commands.command(aliases=["paint"])
    async def generate_image(self, ctx, *, prompt):
        """Generate an image based on the given prompt."""
        if self.pipeline is None:
            await ctx.send("No model loaded. Please load a model first using the 'load' command.")
            return

        async with self.lock:
            async with ctx.typing():
                loop = asyncio.get_event_loop()
                if self.model_quality == "fast":
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        images = await loop.run_in_executor(pool, self.generate_with_sdxl_turbo, prompt)
                elif self.model_quality == "quality":
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        images = await loop.run_in_executor(pool, self.generate_with_stable_cascade, prompt)
                else:
                    await ctx.send("Model not properly loaded. Please reload a model using the 'load' command.")
                    return

                torch.cuda.empty_cache()
                gc.collect()

                # Check if images is not a list, then make it a list
                if not isinstance(images, list):
                    images = [images]

                # Now images is guaranteed to be a list, so we can iterate
                files = []
                if images:
                    for image in images:
                        with io.BytesIO() as binary_img:
                            image.save(binary_img, 'PNG')
                            binary_img.seek(0)
                            files.append(discord.File(binary_img, filename=f'image_{images.index(image)}.png'))
                    await ctx.send(files=files)
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
    chat_cog = True
    if chat_cog and not bot.get_cog("ImageGenerator"):
        #chat_cog = None
        await bot.add_cog(ImageGenerator(bot, chat_cog))
    else:
        print("ImageGenerator cog has already been added.")