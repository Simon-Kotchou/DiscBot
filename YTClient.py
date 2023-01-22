import discord
import youtube_dl
import asyncio
from discord.ext import commands 
from queue import Queue
import sqlite3

#https://stackoverflow.com/questions/66115216/discord-py-play-audio-from-url

youtube_dl.utils.bug_reports_message = lambda: ''

ytdl_config = {
    'format': 'bestaudio/best',
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
    'restrictfilenames': True,
    'noplaylist': True,
    'nocheckcertificate': True,
    'ignoreerrors': False,
    'logtostderr': False,
    'quiet': False,
    'no_warnings': True,
    'default_search': 'auto',
    'source_address': '0.0.0.0'
}

ffmpeg_config = {
    'options': '-vn'
}

yt_client = youtube_dl.YoutubeDL(ytdl_config)

class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.9):
        super().__init__(source, volume)

        self.data = data

        self.title = data.get('title')
        self.url = data.get('url')

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=False):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: yt_client.extract_info(url, download=not stream))

        if 'entries' in data:
            # take first item from a playlist
            data = data['entries'][0]

        filename = data['url'] if stream else yt_client.prepare_filename(data)
        return cls(discord.FFmpegPCMAudio(filename, **ffmpeg_config), data=data)


class Music(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.song_queue = asyncio.Queue(maxsize=50)
        self.current_song = ""
        self.create_table()

    @commands.command(description="joins a voice channel")
    async def join(self, ctx):
        if ctx.author.voice is None or ctx.author.voice.channel is None:
            return await ctx.send('You need to be in a voice channel to use this command!')

        voice_channel = ctx.author.voice.channel
        if ctx.voice_client is None:
            vc = await voice_channel.connect()
        else:
            await ctx.voice_client.move_to(voice_channel)
            vc = ctx.voice_client

    # @commands.command(description="streams music")
    # async def play(self, ctx, *, url):
    #     async with ctx.typing():
    #         player = await YTDLSource.from_url(url, loop=self.bot.loop, stream=True)
    #         ctx.voice_client.play(player, after=lambda e: self.play_next(ctx) if e is None else print('Player error: %s' % e))
    #         self.current_song = player.title
    #         await self.add_to_db(player.title, url, ctx.author.name)
    #     await ctx.send('Now playing: {}'.format(player.title))
    @commands.command(description="streams music")
    async def play(self, ctx, *, url):
        async with ctx.typing():
            player = await YTDLSource.from_url(url, loop=self.bot.loop, stream=True)
            ctx.voice_client.play(player, after=lambda e: self.play_next(ctx) if e is None else print('Player error: %s' % e))
            self.current_song = player.title
            await self.add_to_db(player.title, url, ctx.author.name)
            thumbnail_url = player.data.get('thumbnail')
            if thumbnail_url:
                await ctx.send(f"Now playing: {player.title} with album art {thumbnail_url}")
            else:
                await ctx.send(f'Now playing: {player.title}')

    def play_next(self, ctx):
        async def play_next_async():
            if not self.song_queue.empty():
                next_player = await self.song_queue.get()
                ctx.voice_client.play(next_player, after=lambda e: play_next_async() if e is None else print('Player error: %s' % e))
                self.current_song = next_player.title
                await ctx.send(f'Now playing: {next_player.title}')
            else:
                ctx.voice_client.stop()
        self.bot.loop.create_task(play_next_async())

    @commands.command(description="queue music links")
    async def queue(self, ctx, *, url):
        async with ctx.typing():
            player = await YTDLSource.from_url(url, loop=self.bot.loop, stream=True)
            self.song_queue.put_nowait(player)
            await ctx.send(f'{player.title} has been added to the queue.')

    @commands.command(description="skip the current song")
    async def skip(self, ctx):
        ctx.voice_client.stop()
        if not self.song_queue.empty():
            self.current_song = self.song_queue.get_nowait().title
            await ctx.send(f'Skipping current song {self.current_song}')
            self.play_next(ctx)
        else:
            await ctx.send("No more songs in the queue")

    @commands.command(description="pauses the current song")
    async def pause(self, ctx):
        if ctx.voice_client and ctx.voice_client.is_playing():
            ctx.voice_client.pause()
            await ctx.send('Paused the current song')
        else:
            await ctx.send('Nothing is currently playing')

    @commands.command(description="resumes the currentsong")
    async def resume(self, ctx):
        if ctx.voice_client and ctx.voice_client.is_paused():
            ctx.voice_client.resume()
            await ctx.send('Resumed the current song')
        else:
            await ctx.send('Nothing is currently paused')

    @commands.command(description="shows the recent 5 songs played")
    async def recent(self, ctx):
        conn = sqlite3.connect("music.db")
        c = conn.cursor()
        c.execute("SELECT title, url, user, timestamp FROM music ORDER BY timestamp DESC LIMIT 5")
        result = c.fetchall()
        await ctx.send("Recent 5 songs played:")
        for i, row in enumerate(result):
            await ctx.send(f"{i+1}. {row[0]} - {row[1]} - {row[2]} - {row[3]}")
        c.close()
        conn.close()
    
    @commands.command(description="stops and disconnects the bot from voice")
    async def leave(self, ctx):
        await ctx.voice_client.disconnect()

    @play.before_invoke
    async def ensure_voice(self, ctx):
        if ctx.voice_client is None:
            if ctx.author.voice:
                await ctx.author.voice.channel.connect()
            else:
                await ctx.send("You are not connected to a voice channel.")
                raise commands.CommandError("Author not connected to a voice channel.")
        elif ctx.voice_client.is_playing():
            ctx.voice_client.stop()

    def create_table(self):
        conn = sqlite3.connect("music.db")
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS music 
                     (id INTEGER PRIMARY KEY, title TEXT, url TEXT, user TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()
        conn.close()

    async def add_to_db(self, title, url, user):
        conn = sqlite3.connect("music.db")
        c = conn.cursor()
        c.execute("INSERT INTO music (title, url, user) VALUES (?,?,?)", (title, url, user))
        conn.commit()
        conn.close()

async def setup_yt_client(bot):
   await bot.add_cog(Music(bot))