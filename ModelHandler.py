import torch
from discord.ext import commands
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dataclasses import dataclass
from typing import Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    model_name = "NousResearch/Hermes-2-Theta-Llama-3-8B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            # offload_folder="/tmp/discord_offload",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=nf4_config,
            attn_implementation="flash_attention_2"
        )

        model_handler.loaded_models["chat"] = {
            "model": model,
            "tokenizer": tokenizer,
            "device_map": "auto"
        }
        
        await ctx.send(f"Loaded chat model: {model_name}")
    except Exception as e:
        logger.error(f"Error loading chat model: {e}")
        await ctx.send(f"Failed to load chat model: {str(e)}")

async def load_sdxl_lightning_model(model_handler, ctx):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_8step_unet.safetensors"
    taesd_model = "madebyollin/taesdxl"

    try:
        gpu_indices = await model_handler.allocate_gpus("image", model_handler.available_models[1].num_gpus)
        device = f"cuda:{gpu_indices[0]}"  # Use the first allocated GPU

        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
        ).to(device)
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.vae = AutoencoderTiny.from_pretrained(
            taesd_model, torch_dtype=torch.float16, use_safetensors=True
        ).to(device)

        model_handler.loaded_models["image"] = {
            "pipeline": pipeline,
            "device": device
        }
        await ctx.send(f"Loaded SDXL Lightning model on GPU: {device}")
    except Exception as e:
        logger.error(f"Error loading SDXL Lightning model: {e}")
        await ctx.send(f"Failed to load SDXL Lightning model: {str(e)}")

class ModelHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.loaded_models = {}
        self.model_servers = {}
        self.allocated_gpus = {}
        self.available_models = [
            ModelInfo("chat", "NousResearch/Hermes-2-Theta-Llama-3-8B", 1, load_chat_model),
            ModelInfo("image", "ByteDance/SDXL-Lightning", 1, load_sdxl_lightning_model)
        ]

    async def allocate_gpus(self, model_type: str, num_gpus: int):
        total_gpus = torch.cuda.device_count()
        available_gpus = set(range(total_gpus)) - set(sum(self.allocated_gpus.values(), []))
        
        if len(available_gpus) < num_gpus:
            raise ValueError(f"Not enough available GPUs to load the {model_type} model.")
        
        allocated = sorted(list(available_gpus))[:num_gpus]
        self.allocated_gpus[model_type] = allocated
        return allocated

    async def deallocate_gpus(self, model_type: str):
        if model_type in self.allocated_gpus:
            del self.allocated_gpus[model_type]

    async def load_model(self, ctx, model_type: str):
        if model_type not in [m.model_type for m in self.available_models]:
            await ctx.send(f"Invalid model type. Available models: {', '.join(m.model_type for m in self.available_models)}")
            return

        if model_type in self.loaded_models:
            await ctx.send(f"{model_type} model is already loaded.")
            return

        model_info = next((m for m in self.available_models if m.model_type == model_type), None)
        await model_info.load_function(self, ctx)
        self.model_servers[model_type] = ctx.guild.id

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
        else:
            del model_info["pipeline"]

        await self.deallocate_gpus(model_type)

        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        del self.loaded_models[model_type]
        del self.model_servers[model_type]
        await ctx.send(f"Unloaded {model_type} model.")

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
            gpu_info = model_info.get("device_map", model_info.get("device", "Unknown"))
            info += f"{model_type.capitalize()} Model - Loaded by Server: {server_id}, GPU(s): {gpu_info}\n"
        await ctx.send(info)