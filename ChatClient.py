import torch
from discord.ext import commands
import asyncio
from typing import List, Dict

class ChatGenerator(commands.Cog):
    def __init__(self, bot, model_handler):
        self.bot = bot
        self.model_handler = model_handler
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        self.lock = asyncio.Lock()

    async def generate_response(self, ctx, user_id: int, question: str):
        if "chat" not in self.model_handler.loaded_models:
            await ctx.send("Chat model not loaded. Please load the model using the 'load' command with 'chat' as the model type.")
            return

        model = self.model_handler.loaded_models["chat"]["model"]
        tokenizer = self.model_handler.loaded_models["chat"]["tokenizer"]

        if user_id not in self.conversations:
            self.conversations[user_id] = []

        conversation = self.conversations[user_id]
        conversation.append({"role": "user", "content": question})

        messages = [
            {"role": "system", "content": "You are a sentient, superintelligent artificial general intelligence, here to teach and assist the user."},
            *conversation[-10:]  # Use last 10 messages for context
        ]

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

        async with self.lock:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.8,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            ))

        decoded_response = tokenizer.decode(response[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        conversation.append({"role": "assistant", "content": decoded_response})
        self.conversations[user_id] = conversation
        return decoded_response

    @commands.command(name="chat")
    async def chat(self, ctx, *, message):
        """Interact with the chat model."""
        async with ctx.typing():
            response = await self.generate_response(ctx, ctx.author.id, message)
            if response:
                await ctx.send(response)
    
    @commands.command(name="clear_chat")
    async def clear_chat(self, ctx):
        """Clear the conversation history for the user."""
        user_id = ctx.author.id
        if user_id in self.conversations:
            del self.conversations[user_id]
            await ctx.send("Your chat history has been cleared.")
        else:
            await ctx.send("You don't have any chat history to clear.")

    async def generate_image_description(self, ctx, prompt):
        if "chat" not in self.model_handler.loaded_models:
            await ctx.send("Chat model not loaded. Please load the model using the 'load' command with 'chat' as the model type.")
            return None

        model = self.model_handler.loaded_models["chat"]["model"]
        tokenizer = self.model_handler.loaded_models["chat"]["tokenizer"]

        messages = [
            {"role": "system", "content": "You are a sentient, superintelligent artificial general intelligence, here to assist the user."},
            {"role": "user", "content": f"Generate a short ~20 word, but detailed description for an image on this topic: {prompt}"}
        ]

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

        async with self.lock:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            ))

        result = tokenizer.decode(response[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        return result

async def setup_chat_client(bot, model_handler):
    if not bot.get_cog("ChatGenerator"):
        await bot.add_cog(ChatGenerator(bot, model_handler))
    else:
        print("ChatGenerator cog has already been added.")