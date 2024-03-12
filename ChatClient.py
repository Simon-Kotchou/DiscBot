import torch
from discord.ext import commands
import concurrent.futures
import asyncio
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class ChatGenerator(commands.Cog):
    def __init__(self, bot, model_handler):
        self.bot = bot
        self.model_handler = model_handler
        self.conversations = {}

        self.template = """
        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does not know.
        And please only add the reply, you are the assistant in this case, the human will continue chatting if needed.

        Current conversation:
        {history}
        Human: {input}
        Your reply is: 
        """
        self.prompt = PromptTemplate(input_variables=["history", "input"], template=self.template)

    async def generate_response(self, ctx, user_id, question):
        if "chat" not in self.model_handler.loaded_models:
            await ctx.send("Chat model not loaded. Please load the model using the 'load' command with 'chat' as the model type.")
            return
        
        hf = self.model_handler.loaded_models["chat"]["hf"]

        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationBufferMemory(return_messages=True)

        memory = self.conversations[user_id]

        map_ = RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        chain = (
            map_
            | self.prompt
            | hf
            | StrOutputParser()
        )

        inputs = {"input": question}
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            response = await chain.ainvoke(inputs)
        memory.save_context(inputs, {"output": response})
        self.conversations[user_id] = memory

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
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            prepend = "Please generate a short ~30 word, but detailed description for an image on this topic: "
            prompt = prepend + prompt
            result = await loop.run_in_executor(pool, lambda x: pipe(x), prompt)
            result = result[0]['generated_text'].replace(prompt, "")
        return result

async def setup_chat_client(bot, model_handler):
    if not bot.get_cog("ChatGenerator"):
        await bot.add_cog(ChatGenerator(bot, model_handler))
    else:
        print("ChatGenerator cog has already been added.")
