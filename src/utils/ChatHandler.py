# Built-in imports

# Project imports
from src.data import Color, Config
from src.nlp import PrimitiveModel

# External imports
import discord


class ChatHandler:
    """ Discord interface for the NLP modules """

    def __init__(self, bot):
        """
        Initialize a chat handler interface

        Args:
            bot (BotClient): bot instance
        """
        self.bot = bot

        self.initialize_nlp()

    async def on_message(self, author, message, channel, guild):
        """
        Called automatically after NLP intent is detected

        Args:
            author (discord.Member): message sender
            message (discord.Message): message to parse
            channel (discord.TextChannel): text channel that the message is sent in
            guild (discord.Guild): guild that the message is sent in
        """

        raw_message = message.content
        response = PrimitiveModel.predict(raw_message)
        await channel.send(response, reference=message, mention_author=False)

    def initialize_nlp(self):
        self.bot.log(1, "Loading NLP data... ", print_footer=False)
        PrimitiveModel.load_or_generate_data(force_generate=True)
        self.bot.log(1, "OK!", print_header=False)

        self.bot.log(1, "Training model...")
        PrimitiveModel.create_and_train_model()
        self.bot.log(1, "Training complete! Model is now ready to be used!")
