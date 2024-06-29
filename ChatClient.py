# import torch
# from discord.ext import commands
# import concurrent.futures
# import asyncio
# from operator import itemgetter
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser

# class ChatGenerator(commands.Cog):
#     def __init__(self, bot, model_handler):
#         self.bot = bot
#         self.model_handler = model_handler
#         self.conversations = {}
#         self.lock = asyncio.Lock()

#         self.template = """
#         The following is a friendly conversation between a human and You. 
#         You are talkative and provide lots of specific details from its context. 
#         If you do not know the answer to a question, truthfully says it does not know.
#         And please only add the reply, you are the assistant in this case, the human will continue chatting if needed.

#         Current conversation:
#         {history}
#         Human: {input}
#         Your reply is: 
#         """
#         self.prompt = PromptTemplate(input_variables=["history", "input"], template=self.template)

#     async def generate_response(self, ctx, user_id, question):
#         if "chat" not in self.model_handler.loaded_models:
#             await ctx.send("Chat model not loaded. Please load the model using the 'load' command with 'chat' as the model type.")
#             return
        
#         hf = self.model_handler.loaded_models["chat"]["hf"]

#         if user_id not in self.conversations:
#             self.conversations[user_id] = ConversationBufferMemory(return_messages=True)

#         memory = self.conversations[user_id]

#         map_ = RunnablePassthrough.assign(
#             history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
#         )
#         chain = (
#             map_
#             | self.prompt
#             | hf
#             | StrOutputParser()
#         )

#         inputs = {"input": question}
#         async with self.lock:
#             with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
#                 response = await chain.ainvoke(inputs)
#         memory.save_context(inputs, {"output": response})
#         self.conversations[user_id] = memory

#         return response

#     @commands.command(name="chat")
#     async def chat(self, ctx, *, message):
#         """Interact with the chat model."""
#         async with ctx.typing():
#             response = await self.generate_response(ctx, ctx.author.id, message)
#             if response:
#                 await ctx.send(response)

#     async def generate_image_description(self, ctx, prompt):
#         if "chat" not in self.model_handler.loaded_models:
#             await ctx.send("Chat model not loaded. Please load the model using the 'load' command with 'chat' as the model type.")
#             return None

#         pipe = self.model_handler.loaded_models["chat"]["pipe"]
#         async with self.lock:
#             loop = asyncio.get_event_loop()
#             with concurrent.futures.ThreadPoolExecutor() as pool:
#                 prepend = "Please generate a short ~20 word, but detailed description for an image on this topic: "
#                 prompt = prepend + prompt
#                 result = await loop.run_in_executor(pool, lambda x: pipe(x), prompt)
#                 result = result[0]['generated_text'].replace(prompt, "")
#             return result

# async def setup_chat_client(bot, model_handler):
#     if not bot.get_cog("ChatGenerator"):
#         await bot.add_cog(ChatGenerator(bot, model_handler))
#     else:
#         print("ChatGenerator cog has already been added.")

import torch
from discord.ext import commands
import asyncio
from transformers import pipeline

class ChatGenerator(commands.Cog):
    def __init__(self, bot, model_handler):
        self.bot = bot
        self.model_handler = model_handler
        self.conversations = {}
        self.lock = asyncio.Lock()

    async def generate_response(self, ctx, user_id, question):
        if "chat" not in self.model_handler.loaded_models:
            await ctx.send("Chat model not loaded. Please load the model using the 'load' command with 'chat' as the model type.")
            return

        pipe = self.model_handler.loaded_models["chat"]["pipe"]
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        conversation = self.conversations[user_id]
        conversation.append({"role": "user", "content": question})
        
        prompt = ""
        for message in conversation[-10:]:  # Use last 10 messages for context
            prompt += f"{message['role'].capitalize()}: {message['content']}\n"
        prompt += "Assistant:"

        async with self.lock:
            loop = asyncio.get_event_loop()
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                response = await loop.run_in_executor(None, lambda: pipe(prompt, max_new_tokens=320, repetition_penalty=1.15)[0]['generated_text'])

        response = response.split("Assistant:")[-1].strip()
        conversation.append({"role": "assistant", "content": response})
        self.conversations[user_id] = conversation

        return response

    @commands.command(name="chat")
    async def chat(self, ctx, *, message):
        """Interact with the chat model."""
        async with ctx.typing():
            response = await self.generate_response(ctx, ctx.author.id, message)
            if response:
                await ctx.send(response)

    async def generate_image_description(self, ctx, prompt):
        if "chat" not in self.model_handler.loaded_models:
            await ctx.send("Chat model not loaded. Please load the model using the 'load' command with 'chat' as the model type.")
            return None

        pipe = self.model_handler.loaded_models["chat"]["pipe"]
        async with self.lock:
            loop = asyncio.get_event_loop()
            prepend = "Please generate a short ~20 word, but detailed description for an image on this topic: "
            full_prompt = prepend + prompt
            result = await loop.run_in_executor(None, lambda: pipe(full_prompt, max_new_tokens=100)[0]['generated_text'])
            result = result.split(full_prompt)[-1].strip()
            return result

async def setup_chat_client(bot, model_handler):
    if not bot.get_cog("ChatGenerator"):
        await bot.add_cog(ChatGenerator(bot, model_handler))
    else:
        print("ChatGenerator cog has already been added.")