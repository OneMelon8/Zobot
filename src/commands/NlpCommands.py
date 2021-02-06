# Project imports
from src.utils import StringUtil
from src.utils.CommandHandler import CommandHandler
from src.utils.ReactionHandler import ReactionHandler
from src.data import Config, Emoji, Color
from src.nlp import PrimitiveModel

# External imports
import discord


class ToggleCommandHandler(CommandHandler):
    def __init__(self, bot):
        super().__init__(bot, "toggle", ["t"], "Toggle my NLP chat interface", "", "")

    async def on_command(self, author, command, args, message, channel, guild):
        self.bot.chat_enabled = not self.bot.chat_enabled
        emote = Emoji.UNMUTE if self.bot.chat_enabled else Emoji.MUTE
        await message.add_reaction(emote)
        status = "enabled" if self.bot.chat_enabled else "disabled"
        self.bot.log(1, f"NLP chat interface is now {status}")


class IntentCommandHandler(CommandHandler):
    def __init__(self, bot):
        super().__init__(bot, "intent", ["i", "intents"], "Command to view and modify my NLP intents", f"{Config.BOT_PREFIX}intent <add/info/list/reload> [args...]",
                         f"{Config.BOT_PREFIX}intent add greetings hi there!\n"
                         f"> {Config.BOT_PREFIX}intent info greetings\n"
                         f"> {Config.BOT_PREFIX}intent list\n"
                         f"> {Config.BOT_PREFIX}intent reload")
        self.is_reloading = False

    async def on_command(self, author, command, args, message, channel, guild):
        # Assert there is at least 1 arguments
        if len(args) < 1:
            await self.bot.reply(message, content=f"Invalid arguments! Check out `{Config.BOT_PREFIX}help intent`")
            return

        operation = args[0]
        if operation == "add" or operation == "a":
            if len(args) < 3:
                await self.bot.reply(message, content=f"Invalid arguments! Usage: `{Config.BOT_PREFIX}intent add <intent_name> <utterance...>`")
                return
            elif args[1] not in PrimitiveModel.intents:
                await self.bot.reply(message, embedded=self.get_intent_not_found_embedded(args[1]))
                return
            await self.add_utterance(args[1], " ".join(args[2:]), author, message)
        elif operation == "info" or operation == "i":
            if len(args) < 2:
                await self.bot.reply(message, content=f"Invalid arguments! Usage: `{Config.BOT_PREFIX}intent info <intent_name>`")
                return
            # Show intent information
            if args[1] not in PrimitiveModel.intents:
                response = self.get_intent_not_found_embedded(args[1])
            else:
                response = self.get_intent_info_embedded(args[1])
            await self.bot.reply(message, embedded=response)
        elif operation == "list" or operation == "l":
            # List all intents
            await self.bot.reply(message, embedded=self.get_intent_list_embedded())
        elif operation == "reload" or operation == "r":
            # Reload data and retrain model
            if self.is_reloading:
                await message.add_reaction(Emoji.HOUR_GLASS)
                return
            await self.reload_intents(message)
        else:
            await message.add_reaction(Emoji.QUESTION)
            return

    async def add_utterance(self, intent, utterance, author, reply_message):
        confirmation_message = await self.bot.reply(reply_message, embedded=self.get_add_utterance_confirmation_embedded(intent, utterance))
        await self.bot.react_check(confirmation_message)
        await self.bot.react_cross(confirmation_message)

        async def confirm_add(target_user, user, emote, message, channel, guild):
            # Confirm check emote
            if emote != Emoji.CHECK:
                await confirmation_message.edit(embed=self.get_add_utterance_cancelled_embedded(intent, utterance), mention_author=False)
                return
            # Add utterance to intent json file
            PrimitiveModel.add_utterance(intent, utterance)
            # Edit message
            # TODO: solve the edit-message-mention problem
            await confirmation_message.edit(embed=self.get_add_utterance_successful_embedded(intent, utterance), mention_author=False)

        reaction_handler = ReactionHandler(author, confirmation_message, [Emoji.CHECK, Emoji.CROSS], confirm_add, user_lock=True)
        self.bot.register_reaction_handler(reaction_handler)

    async def reload_intents(self, reply_message):
        self.is_reloading = True

        # Send status: stage 0 -- data reload
        message = await self.bot.reply(reply_message, embedded=self.get_reload_embedded(0))
        # Reload data
        PrimitiveModel.load_or_generate_data(force_generate=True)

        # Send status: stage 1 -- model reload
        await message.edit(embed=self.get_reload_embedded(1))
        # Retrain model
        PrimitiveModel.create_and_train_model()

        # Send status: stage 2 -- done!
        await message.edit(embed=self.get_reload_embedded(2))
        # Set "pending changes" tag to false
        PrimitiveModel.model_changed = False

        self.is_reloading = False

    ###############################
    # EMBEDDED MESSAGE GENERATORS #
    ###############################

    @staticmethod
    def get_add_utterance_confirmation_embedded(intent, utterance):
        embedded = discord.Embed(
            title=f"Confirm add utterance to intent",
            description=f"Are you sure you want to add \"{utterance}\" to intent \"{intent}\"? This operation cannot be undone",
            color=Color.COLOR_NLP
        )
        if PrimitiveModel.model_changed:
            embedded.set_footer(text="* there are some pending changes to the model, reload to see them in action")
        return embedded

    @staticmethod
    def get_add_utterance_successful_embedded(intent, utterance):
        embedded = discord.Embed(
            title=f"Add utterance successful",
            description=f"Successfully added \"{utterance}\" to intent \"{intent}\"",
            color=Color.COLOR_NLP
        )
        if PrimitiveModel.model_changed:
            embedded.set_footer(text="* there are some pending changes to the model, reload to see them in action")
        return embedded

    @staticmethod
    def get_add_utterance_cancelled_embedded(intent, utterance):
        embedded = discord.Embed(
            title=f"Add utterance cancelled",
            description=f"Cancelled adding \"{utterance}\" to intent \"{intent}\"",
            color=Color.COLOR_NLP
        )
        if PrimitiveModel.model_changed:
            embedded.set_footer(text="* there are some pending changes to the model, reload to see them in action")
        return embedded

    @staticmethod
    def get_intent_info_embedded(intent):
        embedded = discord.Embed(
            title=f"Information about intent \"{intent}\"",
            description=f"There is currently a total of **{len(PrimitiveModel.utterances[intent])}** utterances "
                        f"and **{len(PrimitiveModel.responses[intent])}** responses for \"{intent}\"",
            color=Color.COLOR_NLP
        )
        embedded.add_field(name="**Utterances:**", value=f"> {StringUtil.quote_join(PrimitiveModel.utterances[intent])}", inline=False)
        embedded.add_field(name="**Responses:**", value=f"> {StringUtil.quote_join(PrimitiveModel.responses[intent])}", inline=False)
        if PrimitiveModel.model_changed:
            embedded.set_footer(text="* there are some pending changes to the model, reload to see them in action")
        return embedded

    @staticmethod
    def get_intent_not_found_embedded(intent):
        embedded = discord.Embed(
            title=f"Intent \"{intent}\" not found",
            description=f"Try using `{Config.BOT_PREFIX}intent list` to view all intents",
            color=Color.COLOR_NLP
        )
        if PrimitiveModel.model_changed:
            embedded.set_footer(text="* there are some pending changes to the model, reload to see them in action")
        return embedded

    @staticmethod
    def get_intent_list_embedded():
        embedded = discord.Embed(
            title=f"List of intents in my NLP module",
            description=f"There is currently a total of **{len(PrimitiveModel.intents)}** intents",
            color=Color.COLOR_NLP
        )
        embedded.add_field(name="**Intents:**", value=f"> {Config.SEP.join(PrimitiveModel.intents)}", inline=False)
        if PrimitiveModel.model_changed:
            embedded.set_footer(text="* there are some pending changes to the model, reload to see them in action")
        return embedded

    @staticmethod
    def get_reload_embedded(stage):
        descriptions = ["Reloading Data", "Retraining Model", "Complete"]
        descriptions[stage] = f"**{descriptions[stage]}**"
        embedded = discord.Embed(
            title=f"Reloading intent data and retraining models...",
            description=" >> ".join(descriptions),
            color=Color.COLOR_NLP
        )
        return embedded
