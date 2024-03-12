import discord
from YTClient import setup_yt_client
from DiffusionClient import setup_diffusion_client
from ChatClient import setup_chat_client
from ModelHandler import ModelHandler
from discord.ext import commands
import os
from dotenv import load_dotenv
from huggingface_hub import login

if __name__ == "__main__":
    load_dotenv()
    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='/', intents=intents)

    @bot.event
    async def on_ready():
        model_handler = ModelHandler(bot)
        await bot.add_cog(model_handler)
        await setup_yt_client(bot)
        await setup_chat_client(bot, model_handler)
        await setup_diffusion_client(bot, model_handler)

    bot.run(os.getenv('DISCORD_TOKEN'))