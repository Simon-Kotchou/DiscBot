import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
import discord
from discord.ext import commands
from discord import Embed
import asyncio
from tqdm import tqdm
from PIL import Image 
import io
import re
import gc
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import ImageFormatter
import concurrent.futures
from fastchat.conversation import conv_templates, SeparatorStyle, Conversation, get_conv_template
from fastchat.model.model_adapter import load_model, get_conversation_template
from transformers import LogitsProcessorList, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper, TopKLogitsWarper
from collections.abc import Iterable

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False

@torch.inference_mode()
def generate_message(tokenizer, model, params, device,
                     context_len=2048):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 0.7))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", False))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    past_key_values = out = None
    for i in tqdm(range(max_new_tokens)):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            last_token_logits = logits_processor(output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

    del past_key_values
    gc.collect()
    torch.cuda.empty_cache()

    if echo:
        tmp_output_ids = output_ids
        rfind_start = len_prompt
    else:
        tmp_output_ids = output_ids[len(input_ids):]
        rfind_start = 0

    output = tokenizer.decode(
        tmp_output_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )

    partially_stopped = False
    if stop_str:
        if isinstance(stop_str, str):
            pos = output.rfind(stop_str, rfind_start)
            if pos != -1:
                output = output[:pos]
                stopped = True
            else:
                partially_stopped = partial_stop(output, stop_str)
        elif isinstance(stop_str, Iterable):
            for each_stop in stop_str:
                pos = output.rfind(each_stop, rfind_start)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                    break
                else:
                    partially_stopped = partial_stop(output, each_stop)
                    if partially_stopped:
                        break
        else:
            raise ValueError("Invalid stop field type.")

    if not partially_stopped:
        return {
            "text": output,
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": i,
                "total_tokens": len(input_ids) + i,
            },
            "finish_reason": "stop" if stopped else "length",
        }

def generate_code_image(code, lang):
    # Set up the lexer and formatter
    lexer = get_lexer_by_name(lang)
    formatter = ImageFormatter(font_name="Fira Code Regular", font_size=14)

    # Generate a syntax-highlighted image using pygments
    image_data = highlight(code, lexer, formatter)

    # Load the image_data into a PIL Image object
    image_buffer = io.BytesIO(image_data)
    image = Image.open(image_buffer)

    return image

class ChatGenerator(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "../vicuna/vicuna_model"
        self.conv_template = "one_shot"
        
        self.model, self.tokenizer = load_model(
            self.model_name, self.device, num_gpus=2, load_8bit=True, debug=False
        )
        self.lock = asyncio.Lock()
        self.conversations = {}

    def chat_blocking(self, message, conv):
        torch.cuda.empty_cache()

        generate_stream_func = generate_message
        prompt = conv.get_prompt()

        gen_params = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.85,
            "repetition_penalty": 1.0,
            "max_new_tokens": 400,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        output_stream = generate_stream_func(self.tokenizer, self.model, gen_params, self.device)
        outputs = output_stream.get("text", "")
        response = outputs.strip()
        conv.messages[-1][1] = response

        torch.cuda.empty_cache()
        return response

    async def chat_async(self, ctx, message, conv):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(pool, self.chat_blocking, message, conv)
        return response

    async def piped_chat_async(self, ctx, message, conv):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(pool, self.chat_blocking, message, conv)
        return response
    
    @commands.command(name="piped_chat", hidden=True)
    async def piped_chat(self, ctx, *, message):
        #conv = self.pipe_template.copy()
        conv = get_conv_template(self.conv_template)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)

        async with ctx.typing():
            response = await self.piped_chat_async(ctx, message, conv)
            await self.send_large_message(ctx, response)
    
    def split_large_message(self, message, max_chars):
        message_parts = []
        message_words = message.split()
        current_part = ""

        for word in message_words:
            if len(current_part) + len(word) + 1 > max_chars:
                message_parts.append(current_part)
                current_part = word
            else:
                current_part = f"{current_part} {word}".strip()

        message_parts.append(current_part)
        return message_parts
    
    async def send_large_message(self, ctx, message, max_chars=2000):
        """Send a message, splitting it into smaller parts if it exceeds the character limit."""
        code_block_pattern = r"```(\w+)?\n?([\s\S]*?)(```|$)"
        code_blocks = re.findall(code_block_pattern, message)

        # Replace code blocks with placeholders
        for i, (language, code, _) in enumerate(code_blocks):
            message = message.replace(f"```{language}\n{code}```", f"{{{{CODE_BLOCK_{i}}}}}")

        message_parts = self.split_large_message(message, max_chars)

        # Process and send code block images
        for i, (language, code, _) in enumerate(code_blocks):
            image = generate_code_image(code, language)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            image_file = discord.File(buf, filename=f"{language}_code_{i}.png")
            await ctx.send(file=image_file)

        # Send the message parts
        for part in message_parts:
            # Replace the code block placeholders with an empty string
            for i in range(len(code_blocks)):
                part = part.replace(f"{{{{CODE_BLOCK_{i}}}}}", "")

            # Only send the part if it's not an empty string
            if part.strip():
                await ctx.send(part)

    @commands.command(name="chat")
    async def chat(self, ctx, *, message):
        """Start a conversation with the AI."""
        async with self.lock:
            # conv = conv_templates[self.conv_template].copy()
            conv = get_conv_template(self.conv_template)
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            self.conversations[ctx.author.id] = conv

            async with ctx.typing():
                response = await self.chat_async(ctx, message, conv)
                await self.send_large_message(ctx, response)

    @commands.command(name="reply")
    async def reply(self, ctx, *, message):
        if ctx.author.id not in self.conversations:
            await ctx.send("You need to start a conversation first with /chat.")
            return

        def check(msg):
            return msg.author == ctx.author and msg.content.startswith("/reply")

        try:
            async with self.lock:
                conv = self.conversations[ctx.author.id]
                conv.append_message(conv.roles[0], message)
                conv.append_message(conv.roles[1], None)
                async with ctx.typing():
                    response = await self.chat_async(ctx, message, conv)
                    await self.send_large_message(ctx, response)

            reply_message = await self.bot.wait_for("message", check=check, timeout=360)
            await ctx.invoke(self.bot.get_command("reply"), message=reply_message.content[7:])

        except asyncio.TimeoutError:
            await ctx.send(f"{ctx.author.name}'s conversation timed out after 360 seconds. Start a new conversation with /chat.")
            del self.conversations[ctx.author.id]

async def setup_chat_client(bot):
    if not bot.get_cog("ChatGenerator"):
        await bot.add_cog(ChatGenerator(bot))
    else:
        print("ChatGenerator cog has already been added.")