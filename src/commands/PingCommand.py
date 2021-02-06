# Project imports
from src.utils.CommandHandler import CommandHandler
from src.data import Emoji


class PingCommandHandler(CommandHandler):
    """ Command handler for the command "ping" """

    def __init__(self, bot):
        super().__init__(bot, "ping", "", "Check my connection speed to the Discord server", "", "")

    async def on_command(self, author, command, args, message, channel, guild):
        await channel.send(f"{Emoji.PING_PONG} Pong! {int(self.bot.latency * 1000)}ms", reference=message, mention_author=False)
