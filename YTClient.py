import discord
import youtube_dl
import asyncio
from discord.ext import commands 
import sqlite3
import json
import urllib
import requests
import base64

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
    'source_address': '0.0.0.0',
    # 'postprocessors': [{
    #     'key': 'FFmpegExtractAudio',
    #     'preferredcodec': 'mp3',
    #     'preferredquality': '192',
    # },{
    # 'key': 'FFmpegMetadata',
    # 'add_metadata': True,
    # }]
}

ffmpeg_config = {
    'options': '-vn -b:a 320k'
}

yt_client = youtube_dl.YoutubeDL(ytdl_config)

class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.9):
        super().__init__(source, volume)

        self.data = data

        self.title = data.get('title')
        self.url = data.get('url')
        # self.song_title = data.get("song_title")
        # self.artist_name = data.get("artist_name")

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

    @commands.command(description="streams music")
    async def play(self, ctx, *, url, player=None):
        async with ctx.typing():
            if player == None:
                player = await YTDLSource.from_url(url, loop=self.bot.loop, stream=True)
            if ctx.voice_client.is_playing():
                ctx.voice_client.stop()
            ctx.voice_client.play(player, after=lambda e: self.bot.loop.create_task(self.play_next(ctx)) if e is None else print('Player error: %s' % e))
            self.current_song = player.title
            await self.add_to_db(player.title, url, ctx.author.name)
            art_url = await self.get_album_art_spotify(player.title)
            embed = discord.Embed(title=f"Now playing: {player.title}")
            if art_url:
                embed.set_image(url=art_url)
            await ctx.send(embed=embed)

    async def play_next(self, ctx):
        if not self.song_queue.empty():
            next_song, next_url = await self.song_queue.get()
            async with ctx.typing():
                player = await YTDLSource.from_url(next_url, loop=self.bot.loop, stream=True)
                ctx.voice_client.play(player, after=lambda e: self.bot.loop.create_task(self.play_next(ctx)) if e is None else print('Player error: %s' % e))
                self.current_song = player.title
                await self.add_to_db(player.title, next_url, ctx.author.name)
                art_url = await self.get_album_art_spotify(player.title)
                embed = discord.Embed(title=f"Now playing: {player.title}")
                if art_url:
                    embed.set_image(url=art_url)
                await ctx.send(embed=embed)
        else:
            ctx.voice_client.stop()

    def get_client_credentials_token(self):
        client_id = os.getenv('CLIENT_ID')
        client_secret = os.getenv("CLIENT_SECRET")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {base64.b64encode(f'{client_id}:{client_secret}'.encode()).decode()}"
        }

        data = {
            "grant_type": "client_credentials"
        }

        url = "https://accounts.spotify.com/api/token"
        response = requests.post(url, headers=headers, data=data)
        return response.json()["access_token"]

    async def get_album_art_spotify(self, song_title):
        if not song_title:
            return None
        token = self.get_client_credentials_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        query = f"{song_title}"
        query = urllib.parse.quote(query)
        url = f"https://api.spotify.com/v1/search?q={query}&type=track"
        response = requests.get(url, headers=headers)
        data = json.loads(response.text)
        try:
            art_url = data.get("tracks", {}).get("items", [{}])[0].get("album", {}).get("images", [{}])[0].get("url")
        except FileNotFoundError:
            art_url = None
        return art_url

    @commands.command(description="generates spotify recommendations")
    async def recommend(self, ctx):
        async with ctx.typing():
            recommendations = await self.generate_recommendations()
            if not recommendations:
                return await ctx.send("No recommendations could be generated.")
            message = "Here are your recommendations:\n"
            for i, track in enumerate(recommendations):
                message += f"{i+1}. {track['name']} by {track['artists'][0]['name']}\n"
            await ctx.send(message)

    async def generate_recommendations(self):
        # Get the access token
        token = self.get_client_credentials_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Get the past 10 songs played from the database
        past_songs = await self.get_past_songs(10)
        if not past_songs:
            return None

        # Get the Spotify IDs of the past songs
        track_ids = []
        for song in past_songs:
            track_id = await self.get_track_id(song[0], token=token)
            if track_id:
                track_ids.append(track_id)

        if not track_ids:
            return None

        # Get the recommendations from Spotify
        url = f"https://api.spotify.com/v1/recommendations?seed_tracks={','.join(track_ids)}"
        response = requests.get(url, headers=headers)
        data = json.loads(response.text)
        recommendations = data.get("tracks", [])

        return recommendations

    async def get_track_id(self, title, token=None):
        # Search for the song on Spotify and return its ID
        if token == None:
            token = self.get_client_credentials_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        query = f"{title}"
        query = urllib.parse.quote(query)
        url = f"https://api.spotify.com/v1/search?q={query}&type=track"
        response = requests.get(url, headers=headers)
        data = json.loads(response.text)
        try:
            track_id = data.get("tracks", {}).get("items", [{}])[0].get("id")
        except FileNotFoundError:
            track_id = None
        return track_id

    @commands.command(description="queue music links")
    async def queue(self, ctx, *, url):
        async with ctx.typing():
            player = await YTDLSource.from_url(url, loop=self.bot.loop, stream=True)
            await self.song_queue.put((player, url))
            await ctx.send(f'{player.title} has been added to the queue.')

    @commands.command(description="skip the current song")
    async def skip(self, ctx):
        ctx.voice_client.stop()
        if not self.song_queue.empty():
            await ctx.send(f'Skipping current song {self.current_song}')
            self.bot.create_task(self.play_next(ctx))
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

    async def get_past_songs(self, limit):
        conn = sqlite3.connect('music.db')
        cursor = conn.cursor()
        cursor.execute("SELECT title, url FROM music ORDER BY id DESC LIMIT ?", (limit,))
        past_songs = cursor.fetchall()
        conn.close()
        return past_songs

async def setup_yt_client(bot):
   await bot.add_cog(Music(bot))