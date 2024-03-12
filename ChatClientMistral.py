import discord
import torch
from discord.ext import commands
import concurrent.futures
import asyncio
from operator import itemgetter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

class ChatGenerator(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.conversations = {}

        # Initialize the model and chain
        model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          offload_folder = "/tmp/discord_offload",
                                                          #use_flash_attention_2=True,
                                                          quantization_config=nf4_config,
                                                          max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')
        self.pipe = pipeline("text-generation", 
                             model=self.model,
                             tokenizer=self.tokenizer,
                             #device_map="auto",
                             device_map="cuda:0",
                             max_new_tokens=512, 
                             repetition_penalty=1.15)
        self.hf = HuggingFacePipeline(pipeline=self.pipe)

        self.template = """
        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does not know.
        And please only add the reply, you are the assistant in this case, the human will continue chatting if needed.

        Current conversation:
        {history}
        You should answer this question: {input}
        """
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
    
    async def piped_chat(self, ctx, prompt):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            prepend = "Please generate a short ~30 word, but detailed description for an image on this topic: "
            prompt = prepend + prompt
            result = await loop.run_in_executor(pool, lambda x: self.pipe(x), prompt)
            result = result[0]['generated_text'].replace(prompt, "")
        return result

async def setup_chat_client(bot):
    if not bot.get_cog("ChatGenerator"):
        await bot.add_cog(ChatGenerator(bot))
    else:
        print("ChatGenerator cog has already been added.")