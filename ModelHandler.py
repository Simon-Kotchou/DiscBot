import torch
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
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
    required_memory: int = 0

async def load_chat_model(model_handler, ctx, gpu_indices):
    gpu_index = gpu_indices[0]
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get the total memory of the specific GPU
    gpu_total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
    max_memory = f'{int(gpu_total_memory / 1024**3) - 2}GB'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        offload_folder="/tmp/discord_offload",
        quantization_config=nf4_config,
        max_memory=max_memory
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=f"cuda:{gpu_index}",
        max_new_tokens=320,
        repetition_penalty=1.15
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    model_handler.loaded_models["chat"] = {
        "model": model,
        "tokenizer": tokenizer,
        "pipe": pipe,
        "hf": hf,
        "gpu_index": gpu_index
    }
    await ctx.send(f"Loaded chat model on GPU {gpu_index}.")

async def load_sdxl_lightning_model(model_handler, ctx, gpu_indices):
    gpu_index = gpu_indices[0]
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(f"cuda:{gpu_index}", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=f"cuda:{gpu_index}"))

    pipeline = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(f"cuda:{gpu_index}")
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    model_handler.loaded_models["image"] = {
        "pipeline": pipeline,
        "gpu_index": gpu_index
    }
    await ctx.send(f"Loaded SDXL Lightning model on GPU {gpu_index}.")

class ModelHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.loaded_models = {}
        self.model_servers = {}
        self.gpu_info = self.get_gpu_info()
        self.allocated_gpus = []  # Initialize allocated_gpus as an empty list
        self.available_models = [
            ModelInfo("chat", "mistralai/Mistral-7B-Instruct-v0.2", 1, load_chat_model, required_memory=7064451072),
            ModelInfo("image", "ByteDance/SDXL-Lightning", 1, load_sdxl_lightning_model, required_memory=7065827840)
        ]

    def get_gpu_info(self):
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_reserved": torch.cuda.memory_reserved(i)
            })
        return gpu_info

    def get_available_gpus(self, required_memory):
        available_gpus = []
        for gpu in self.gpu_info:
            if gpu["index"] not in self.allocated_gpus:  # Check if the GPU is not already allocated
                free_memory = gpu["memory_total"] - gpu["memory_allocated"]
                if free_memory >= required_memory:
                    available_gpus.append(gpu["index"])
        return available_gpus

    async def load_model(self, ctx, model_type: str):
        if model_type not in [m.model_type for m in self.available_models]:
            await ctx.send(f"Invalid model type. Available models: {', '.join(m.model_type for m in self.available_models)}")
            return

        if model_type in self.loaded_models:
            await ctx.send(f"{model_type} model is already loaded.")
            return

        model_info = next((m for m in self.available_models if m.model_type == model_type), None)
        required_memory = model_info.required_memory if hasattr(model_info, 'required_memory') else 0
        available_gpus = self.get_available_gpus(required_memory)

        if len(available_gpus) >= model_info.num_gpus:
            gpu_indices = available_gpus[:model_info.num_gpus]
            self.allocated_gpus.extend(gpu_indices)  # Allocate the selected GPUs
            await model_info.load_function(self, ctx, gpu_indices)
            self.model_servers[model_type] = ctx.guild.id
        else:
            await ctx.send(f"Not enough available GPUs to load the {model_type} model.")

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

        gpu_indices = model_info["gpu_index"] if isinstance(model_info["gpu_index"], list) else [model_info["gpu_index"]]
        for gpu_index in gpu_indices:
            self.allocated_gpus.remove(gpu_index)  # Deallocate the GPUs

        torch.cuda.empty_cache()
        del self.loaded_models[model_type]
        del self.model_servers[model_type]
        await ctx.send(f"Unloaded {model_type} model.")

    @commands.command(name="gpu_info")
    async def gpu_info_command(self, ctx):
        self.gpu_info = self.get_gpu_info()  # Update GPU information
        info = "GPU Information:\n"
        for gpu in self.gpu_info:
            info += f"GPU {gpu['index']}: {gpu['name']}, Memory: {gpu['memory_allocated']} / {gpu['memory_total']}\n"
        await ctx.send(info)

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
            if model_type == "image_quality":
                info += f"{model_type.capitalize()} Model - GPU(s): {model_info['gpu_indices']}, Loaded by Server: {server_id}\n"
            else:
                info += f"{model_type.capitalize()} Model - GPU: {model_info['gpu_index']}, Loaded by Server: {server_id}\n"
        await ctx.send(info)