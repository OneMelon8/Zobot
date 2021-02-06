# Project imports
from src.utils.CommandHandler import CommandHandler
from src.data import Color, Config, Emoji

# External imports
import discord


class HelpCommandHandler(CommandHandler):
    """ Command handler for the command "help" """

    def __init__(self, bot):
        super().__init__(bot, "help", ["?"], "Show help message for a command", "/help [command]", "/help ping")

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
                await message.add_reaction(Emoji.QUESTION)
                reply_embedded = self.get_unknown_command_embedded()
            else:
                reply_embedded = self.get_command_help_embedded(handler)
        # Unknown format, reply with question mark
        else:
            await message.add_reaction(Emoji.QUESTION)
            return

        await channel.send(embed=reply_embedded, reference=message, mention_author=False)

    def get_general_help_embedded(self):
        embedded = discord.Embed(
            title=f"List of available commands",
            description=f"Here's how to use my commands: `/<command> [arguments...]`",
            color=Color.COLOR_HELP
        )
        embedded.add_field(name="**List of commands:**", value=f"> {Config.SEP.join(handler.command for handler in self.bot.command_handlers)}", inline=False)
        embedded.set_footer(text="For more information, check out '/help [command]'")
        return embedded

    @staticmethod
    def get_command_help_embedded(handler):
        return handler.get_help_embedded()

    def get_unknown_command_embedded(self):
        embedded = discord.Embed(
            title=f"Unknown command \"{self.command}\"",
            description=f"That is not a valid command, check out a list of commands with `{Config.BOT_PREFIX}help`",
            color=Color.COLOR_HELP
        )
        return embedded
