# Built-in imports

# Project imports
from src.data import Color, Config, Emoji
from src.nlp import PrimitiveModel
from src.utils.ReactionHandler import ReactionHandler

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
        response, confidence, results = PrimitiveModel.predict(raw_message)

        # if confidence < Config.NLP_CONFIDENCE_THRESHOLD:
        #     return
        result_message = await channel.send(response, reference=message, mention_author=False)
        await result_message.add_reaction(Emoji.MAGNIFYING_GLASS)

        async def on_react(target_user, user, emote, message, channel, guild):
            # Confirm emote
            if emote != Emoji.MAGNIFYING_GLASS:
                return
            await result_message.edit(embed=self.get_nlp_results_embedded(results), mention_author=False)

        reaction_handler = ReactionHandler(author, result_message, [Emoji.MAGNIFYING_GLASS], on_react)
        self.bot.register_reaction_handler(reaction_handler)

    def initialize_nlp(self):
        self.bot.log(1, "Loading NLP data... ", print_footer=False)
        PrimitiveModel.load_or_generate_data(force_generate=True)
        self.bot.log(1, "OK!", print_header=False)

        self.bot.log(1, "Training model...")
        PrimitiveModel.create_and_train_model()
        self.bot.log(1, "Training complete! Model is now ready to be used!")

    @staticmethod
    def get_nlp_results_embedded(results):
        results = sorted(results.items(), key=lambda a: a[1], reverse=True)

        embedded = discord.Embed(
            title=f"Detailed results of this response",
            description=f"Best matching intent is \"{results[0][0]}\" with {results[0][1] * 100:05.2f}% confidence",
            color=Color.COLOR_NLP
        )

        detail_string = "```"
        for intent, confidence in results:
            detail_string += f"{intent:15s} ({confidence * 100:05.2f}%)\n"
        detail_string += "```"

        embedded.add_field(name="**Detailed results:**", value=detail_string, inline=False)
        return embedded
