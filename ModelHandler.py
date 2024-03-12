import torch
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from diffusers import AutoPipelineForText2Image, StableCascadePriorPipeline, StableCascadeDecoderPipeline
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

class ModelHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.server_models = {}
        self.gpu_info = self.get_gpu_info()
        self.available_models = [
            ModelInfo("chat", "mistralai/Mistral-7B-Instruct-v0.2", 1, self.load_chat_model),
            ModelInfo("image_fast", "stabilityai/sdxl-turbo", 1, self.load_sdxl_turbo_model),
            ModelInfo("image_quality", "stabilityai/stable-cascade", 2, self.load_stable_cascade_model)
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

    def get_available_gpus(self):
        available_gpus = []
        for gpu in self.gpu_info:
            if gpu["memory_allocated"] == 0 and gpu["memory_reserved"] == 0:
                available_gpus.append(gpu["index"])
        return available_gpus

    async def load_model(self, ctx, model_type: str):
        server_id = ctx.guild.id
        if server_id not in self.server_models:
            self.server_models[server_id] = {}

        model_info = next((m for m in self.available_models if m.model_type == model_type), None)
        if model_info is None:
            await ctx.send(f"Invalid model type. Available models: {', '.join(m.model_type for m in self.available_models)}")
            return

        if model_info.model_type in self.server_models[server_id]:
            await ctx.send(f"{model_info.model_type} model is already loaded for this server.")
            return

        available_gpus = self.get_available_gpus()
        if len(available_gpus) >= model_info.num_gpus:
            gpu_indices = available_gpus[:model_info.num_gpus]
            await model_info.load_function(ctx, server_id, gpu_indices)
        else:
            await ctx.send(f"Not enough available GPUs to load the {model_info.model_type} model.")

    async def load_chat_model(self, ctx, server_id, gpu_indices: List[int]):
        gpu_index = gpu_indices[0]
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            offload_folder="/tmp/discord_offload",
            quantization_config=nf4_config,
            max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=f"cuda:{gpu_index}",
            max_new_tokens=512,
            repetition_penalty=1.15
        )
        hf = HuggingFacePipeline(pipeline=pipe)

        self.server_models[server_id]["chat"] = {
            "model": model,
            "tokenizer": tokenizer,
            "pipe": pipe,
            "hf": hf,
            "gpu_index": gpu_index
        }
        await ctx.send(f"Loaded chat model on GPU {gpu_index} for this server.")

    async def load_sdxl_turbo_model(self, ctx, server_id, gpu_indices: List[int]):
        gpu_index = gpu_indices[0]
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(f"cuda:{gpu_index}")
        self.server_models[server_id]["diffusion"] = {
            "quality": "fast",
            "pipeline": pipeline,
            "gpu_index": gpu_index
        }
        await ctx.send(f"Loaded diffusion model with fast quality on GPU {gpu_index} for this server.")

    async def load_stable_cascade_model(self, ctx, server_id, gpu_indices: List[int]):
        prior_gpu_index, decoder_gpu_index = gpu_indices
        pipeline = {
            "prior": StableCascadePriorPipeline.from_pretrained(
                "stabilityai/stable-cascade-prior",
                torch_dtype=torch.bfloat16
            ).to(f"cuda:{prior_gpu_index}"),
            "decoder": StableCascadeDecoderPipeline.from_pretrained(
                "stabilityai/stable-cascade",
                torch_dtype=torch.float16,
                revision="refs/pr/17"
            ).to(f"cuda:{decoder_gpu_index}")
        }
        self.server_models[server_id]["diffusion"] = {
            "quality": "quality",
            "pipeline": pipeline,
            "gpu_indices": gpu_indices
        }
        await ctx.send(f"Loaded diffusion model with quality settings on GPUs {gpu_indices} for this server.")

    async def unload_model(self, ctx, model_type: str):
        server_id = ctx.guild.id
        if server_id in self.server_models and model_type in self.server_models[server_id]:
            del self.server_models[server_id][model_type]
            await ctx.send(f"Unloaded {model_type} model for this server.")
        else:
            await ctx.send(f"No {model_type} model loaded for this server.")

    @commands.command(name="gpu_info")
    async def gpu_info_command(self, ctx):
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
        server_id = ctx.guild.id
        if server_id in self.server_models:
            info = f"Loaded Models for Server {server_id}:\n"
            for model_type, model_info in self.server_models[server_id].items():
                if model_type == "diffusion":
                    info += f"{model_type.capitalize()} Model - Quality: {model_info['quality']}, GPU(s): {model_info['gpu_indices']}\n"
                else:
                    info += f"{model_type.capitalize()} Model - GPU: {model_info['gpu_index']}\n"
        else:
            info = "No models loaded for this server."
        await ctx.send(info)