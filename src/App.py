# Built-in imports
import os

# Project imports
from src.Bot import BotClient
from src.commands import NlpCommands, UtilityCommands
from src.utils.ChatHandler import ChatHandler

# External imports
import discord
from dotenv import load_dotenv

# Load simulated environment

load_dotenv()

# Get Discord token from the environment
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Create intent
intent = discord.Intents.default()
intent.members = True

# Create and start the client
client = BotClient(intents=intent)

# Register commands
client.register_command_handler(NlpCommands.ToggleCommandHandler(client))
client.register_command_handler(NlpCommands.IntentCommandHandler(client))
client.register_command_handler(UtilityCommands.PingCommandHandler(client))
client.register_command_handler(UtilityCommands.HelpCommandHandler(client))

# Register NLP chat handler
client.register_chat_handler(ChatHandler(client))

client.run(BOT_TOKEN)
