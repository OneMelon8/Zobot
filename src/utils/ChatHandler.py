# Built-in imports
import random
import pickle

# Project imports
from src.data import Color, Config

# External imports
import discord
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np


class ChatHandler:
    """ Discord interface for the NLP modules """

    def __init__(self, bot):
        """
        Initialize a chat handler interface

        Args:
            bot (BotClient): bot instance
        """
        self.bot = bot

        self.stemmer = LancasterStemmer()

    async def on_message(self, author, message, channel, guild):
        """
        Called automatically after NLP intent is detected

        Args:
            author (discord.Member): message sender
            message (discord.Message): message to parse
            channel (discord.TextChannel): text channel that the message is sent in
            guild (discord.Guild): guild that the message is sent in
        """

        await channel.send("WIP", reference=message, mention_author=False)

    def load_model(self):
        pass

    def bag_of_words(self, message, word_bank):
        pass
