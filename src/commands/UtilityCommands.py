# Project imports
from src.utils.CommandHandler import CommandHandler
from src.data import Color, Config, Emoji

# External imports
import discord


class HelpCommandHandler(CommandHandler):
    def __init__(self, bot):
        super().__init__(bot, "help", ["?"], "Show help message for a command", f"{Config.BOT_PREFIX}help [command]", f"{Config.BOT_PREFIX}help ping")

    async def on_command(self, author, command, args, message, channel, guild):
        # Help in general
        if len(args) == 0:
            reply_embedded = self.get_general_help_embedded()
        # Help for specific command
        elif len(args) == 1:
            # Find target command
            handler = None
            for loop in self.bot.command_handlers:
                if args[0] == loop.command or args[0] in loop.aliases:
                    handler = loop
                    break

            # Not found -- unknown command
            if handler is None:
                reply_embedded = self.get_unknown_command_embedded(args[0])
            else:
                reply_embedded = self.get_command_help_embedded(handler)
        # Unknown format, reply with question mark
        else:
            await self.bot.react_unknown(message)
            return

        await self.bot.reply(message, embedded=reply_embedded)

    def get_general_help_embedded(self):
        embedded = discord.Embed(
            title=f"List of available commands",
            description=f"Here's how to use my commands: `{Config.BOT_PREFIX}<command> [arguments...]`",
            color=Color.COLOR_HELP
        )
        embedded.add_field(name="**List of commands:**", value=f"> {Config.SEP.join(handler.command for handler in self.bot.command_handlers)}", inline=False)
        embedded.set_footer(text=f"For more information, check out '{Config.BOT_PREFIX}help [command]'")
        return embedded

    @staticmethod
    def get_command_help_embedded(handler):
        return handler.get_help_embedded()

    @staticmethod
    def get_unknown_command_embedded(command):
        embedded = discord.Embed(
            title=f"Unknown command \"{command}\"",
            description=f"That is not a valid command, check out a list of commands with `{Config.BOT_PREFIX}help`",
            color=Color.COLOR_HELP
        )
        return embedded


class PingCommandHandler(CommandHandler):
    def __init__(self, bot):
        super().__init__(bot, "ping", [], "Check my connection speed to the Discord server", "", "")

    async def on_command(self, author, command, args, message, channel, guild):
        await self.bot.reply(message, content=f"{Emoji.PING_PONG} Pong! {int(self.bot.latency * 1000)}ms")
