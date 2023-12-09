from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.utils import pt_to_pil
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
        self.device_1 = "cuda:0"
        self.device_2 = "cuda:1"
        #self.repo_id_gen = "stabilityai/stable-diffusion-2"
        self.repo_id_gen = "DeepFloyd/IF-I-XL-v1.0"
        self.stage_2_id = "DeepFloyd/IF-II-L-v1.0"
        self.upscaler = "stabilityai/stable-diffusion-x4-upscaler"
        #self.pipe_gen = DiffusionPipeline.from_pretrained(self.repo_id_gen, torch_dtype=torch.float16, revision="fp16")
        self.pipe_gen = DiffusionPipeline.from_pretrained(self.repo_id_gen, variant="fp16", torch_dtype=torch.float16)
        self.pipe_gen.enable_model_cpu_offload(0)
        self.stage_2 = DiffusionPipeline.from_pretrained(self.stage_2_id, text_encoder=None, variant="fp16", torch_dtype=torch.float16)
        self.stage_2.enable_model_cpu_offload(1)
        self.stage_3 = DiffusionPipeline.from_pretrained(self.upscaler, torch_dtype=torch.float16)
        self.stage_3.enable_model_cpu_offload(1)

        # self.pipe_gen.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe_gen.scheduler.config)
        # self.pipe_gen = self.pipe_gen.to(self.device_1)

        # self.repo_id_edit = "timbrooks/instruct-pix2pix"
        # self.pipe_edit = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.repo_id_edit, torch_dtype=torch.float16).to(self.device_2)

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
        torch.cuda.empty_cache()
        with torch.no_grad():
            #with torch.cuda.device(0):
            #image = self.pipe_gen(prompt, guidance_scale=9, num_inference_steps=300).images[0]
            generator = torch.manual_seed(1)

            # text embeds
            prompt_embeds, negative_embeds = self.pipe_gen.encode_prompt(prompt)
            torch.cuda.empty_cache()

            # stage 1
            image = self.pipe_gen(
                prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, num_inference_steps=250, output_type="pt"
            ).images
            torch.cuda.empty_cache()
            # stage 2
            image = self.stage_2(
                image=image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                generator=generator,
                num_inference_steps=100,
                output_type="pt",
            ).images
            del prompt_embeds 
            del negative_embeds
            gc.collect()
            torch.cuda.empty_cache()
            # stage 3
            image = self.stage_3(prompt=prompt, image=image, noise_level=100, generator=generator, num_inference_steps=100).images[0]
            torch.cuda.empty_cache()
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
    
    def edit_image_blocking(self, prompt, image):
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.device(1):
                image = self.pipe_edit(prompt, image=image, num_inference_steps=300, image_guidance_scale=1.5, guidance_scale=7).images[0]
        torch.cuda.empty_cache()
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
        async with self.lock:
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

    async def execute_piped_commands(self, ctx, commands, debug=True):
        result = None
        for command_str in commands:
            command, *args = command_str.split()

            if command == "paint":
                result = await self.generate_image_async(ctx, ' '.join(args))
            elif command == "edit":
                if result:
                    result = await self.edit_image_async(ctx, ' '.join(args), result)
                else:
                    # No input to edit command
                    return False
            elif command == "chat":
                conv = self.chat_cog.pipe_template.copy()
                if result:
                    prompt = f"{result} {' '.join(args)}"
                else:
                    prompt = ' '.join(args)
                result = await self.chat_cog.piped_chat_async(ctx, prompt, conv)
                if debug:
                    print(f"Chat Output: {result}")
            else:
                # Unknown command
                return False

        if isinstance(result, PIL.Image.Image):
            with io.BytesIO() as binary_img:
                result.save(binary_img, 'PNG')
                binary_img.seek(0)
                file = discord.File(binary_img, filename='result.png')
                await ctx.send(file=file)
        else:
            await ctx.send(result)

        return True

    @commands.command(name="pipe")
    async def pipe(self, ctx, *, piped_commands):
        commands = piped_commands.split("|")
        success = await self.execute_piped_commands(ctx, commands)
        if not success:
            await ctx.send("Invalid piped command.")


async def setup_diffusion_client(bot):
    #chat_cog = bot.get_cog("ChatGenerator")
    chat_cog = True
    if chat_cog and not bot.get_cog("ImageGenerator"):
        chat_cog = None
        await bot.add_cog(ImageGenerator(bot, chat_cog))
    else:
        print("ImageGenerator cog has already been added.")