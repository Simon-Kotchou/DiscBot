import torch
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dataclasses import dataclass
from typing import Callable, List

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

@dataclass
class ModelInfo:
    model_type: str
    model_name: str
    num_gpus: int
    load_function: Callable

async def load_chat_model(model_handler, ctx):
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        offload_folder="/tmp/discord_offload",
        quantization_config=nf4_config
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device_map="auto",
        max_new_tokens=320,
        repetition_penalty=1.15
    )
    pipe.enable_model_cpu_offload()
    hf = HuggingFacePipeline(pipeline=pipe)

    model_handler.loaded_models["chat"] = {
        "model": model,
        "tokenizer": tokenizer,
        "pipe": pipe,
        "hf": hf
    }
    await ctx.send("Loaded chat model.")

async def load_sdxl_lightning_model(model_handler, ctx):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"
    taesd_model = "madebyollin/taesdxl"

    unet = UNet2DConditionModel.from_config(base, subfolder="unet", torch_dtype=torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
    pipeline.vae = AutoencoderTiny.from_pretrained(
        taesd_model, torch_dtype=torch.float16, use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()
    model_handler.loaded_models["image"] = {
        "pipeline": pipeline
    }
    await ctx.send("Loaded SDXL Lightning model.")

class ModelHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.loaded_models = {}
        self.model_servers = {}
        self.allocated_gpus = set()
        self.available_models = [
            ModelInfo("chat", "mistralai/Mistral-7B-Instruct-v0.2", 2, load_chat_model),
            ModelInfo("image", "ByteDance/SDXL-Lightning", 2, load_sdxl_lightning_model)
        ]

    async def load_model(self, ctx, model_type: str):
        if model_type not in [m.model_type for m in self.available_models]:
            await ctx.send(f"Invalid model type. Available models: {', '.join(m.model_type for m in self.available_models)}")
            return

        if model_type in self.loaded_models:
            await ctx.send(f"{model_type} model is already loaded.")
            return

        model_info = next((m for m in self.available_models if m.model_type == model_type), None)
        num_gpus = model_info.num_gpus

        if len(self.allocated_gpus) + num_gpus > torch.cuda.device_count():
            await ctx.send(f"Not enough available GPUs to load the {model_type} model.")
            return

        gpu_indices = list(range(len(self.allocated_gpus), len(self.allocated_gpus) + num_gpus))
        self.allocated_gpus.update(gpu_indices)
        await model_info.load_function(self, ctx)
        self.model_servers[model_type] = ctx.guild.id
        await ctx.send(f"Loaded {model_type} model on GPUs: {gpu_indices}")

    async def unload_model(self, ctx, model_type: str):
        if model_type not in self.loaded_models:
            await ctx.send(f"No {model_type} model loaded.")
            return

        if self.model_servers.get(model_type) != ctx.guild.id:
            await ctx.send(f"The {model_type} model was not loaded by this server. It cannot be unloaded.")
            return

        model_info = self.loaded_models[model_type]
        if model_type == "chat":
            del model_info["model"]
            del model_info["tokenizer"]
            del model_info["pipe"]
            del model_info["hf"]
        else:
            del model_info["pipeline"]

        num_gpus = next((m.num_gpus for m in self.available_models if m.model_type == model_type), 0)
        gpu_indices = list(range(len(self.allocated_gpus) - num_gpus, len(self.allocated_gpus)))
        self.allocated_gpus.difference_update(gpu_indices)

        torch.cuda.empty_cache()
        del self.loaded_models[model_type]
        del self.model_servers[model_type]
        await ctx.send(f"Unloaded {model_type} model from GPUs: {gpu_indices}")

    @commands.command(name="load")
    async def load_command(self, ctx, model_type: str):
        await self.load_model(ctx, model_type)

    @commands.command(name="unload")
    async def unload_command(self, ctx, model_type: str):
        await self.unload_model(ctx, model_type)

    @commands.command(name="loaded_models")
    async def loaded_models_command(self, ctx):
        if not self.loaded_models:
            await ctx.send("No models loaded.")
            return

        info = "Loaded Models:\n"
        for model_type, model_info in self.loaded_models.items():
            server_id = self.model_servers.get(model_type)
            info += f"{model_type.capitalize()} Model - Loaded by Server: {server_id}\n"
        await ctx.send(info)