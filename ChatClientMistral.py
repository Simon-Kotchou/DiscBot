import discord
import torch
from discord.ext import commands
import concurrent.futures
from operator import itemgetter
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class ChatGenerator(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.conversations = {}

        # Initialize the model and chain
        model_name = 'Intel/neural-chat-7b-v3-1'
        self.hf = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device_map="auto",
            verbose=True,
            pipeline_kwargs={"max_new_tokens": 256, "repetition_penalty": 1.15}, 
        )
        self.template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {history}
        Human: {input}
        AI Assistant:"""
        self.prompt = PromptTemplate(input_variables=["history", "input"], template=self.template)

    async def generate_response(self, user_id, question):
        # Retrieve or create a conversation context
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationBufferMemory(return_messages=True)

        memory = self.conversations[user_id]

        map_ = RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        chain = (
            map_
            | self.prompt 
            | self.hf 
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
            response = await self.generate_response(ctx.author.id, message)
            await ctx.send(response)

async def setup_chat_client(bot):
    if not bot.get_cog("ChatGenerator"):
        await bot.add_cog(ChatGenerator(bot))
    else:
        print("ChatGenerator cog has already been added.")