import discord
import pymorphy2

TOKEN = 'MTEwMTgzNzMwODk3NTY1Njk2MQ.GbP8gS.6ErX743Orc29uKEX1tanA8HCauvGwF9DTu9_44'
client = discord.Client(intents=discord.Intents(guilds=True, messages=True))
morph = pymorphy2.MorphAnalyzer()


@client.event
async def on_ready():
    print('Bot is ready')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    elif message.content.startswith('#!numerals'):
        words = message.content.split()[1:]
        num = words[0]
        parsed_word = morph.parse(words[1])[0]
        if not parsed_word.is_noun:
            await message.channel.send('Это не существительное')
        elif parsed_word.tag.gender is None:
            await message.channel.send('Не удалось определить род существительного')
        else:
            inflected_word = parsed_word.inflect({'gent', 'plural'})
            if inflected_word is None:
                await message.channel.send('Не удалось согласовать существительное с числительным')
            else:
                inflected_word = inflected_word.word
                await message.channel.send(f"{num} {inflected_word}")


    elif message.content.startswith('#lalive'):
        text = message.content.split(maxsplit=1)[1]
        words = text.split()
        if morph.parse(words[0])[0].tag.POS == 'NOUN':
            parsed_word = morph.parse(words[0])[0]
            if parsed_word.tag.animacy == 'anim':
                await message.channel.send('Живой')
            else:
                await message.channel.send('Не живой')
        else:
            await message.channel.send('Не существительное')

    elif message.content.startswith('#!noun'):
        text = message.content.split(maxsplit=1)[1]
        words = text.split()
        if len(words) != 3:
            await message.channel.send('Invalid input')
            return
        if morph.parse(words[0])[0].tag.POS != 'NOUN':
            await message.channel.send('Not a noun')
            return
        try:
            parsed_word = morph.parse(words[0])[0]
            inflected_word = parsed_word.inflect({words[1], words[2]})
            if inflected_word:
                await message.channel.send(inflected_word.word)
            else:
                await message.channel.send('Invalid input')
        except ValueError:
            await message.channel.send('Invalid input')

    elif message.content.startswith('#linf'):
        text = message.content.split(maxsplit=1)[1]
        words = text.split()
        parsed_word = morph.parse(words[0])[0]
        await message.channel.send(parsed_word.normal_form)

    elif message.content.startswith('#!morph'):
        text = message.content.split(maxsplit=1)[1]
        words = text.split()
        parsed_word = morph.parse(words[0])[0]
        max_score = parsed_word.score
        max_word = parsed_word.word
        for parsed in morph.parse(words[0]):
            if parsed.score > max_score:
                max_score = parsed.score
                max_word = parsed.word
        await message.channel.send(f'{max_word}, {max_score}')

client.run(TOKEN)
