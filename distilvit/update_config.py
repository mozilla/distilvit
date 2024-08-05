"""
Script to add the bad words ids to the generate_config file
"""
from transformers import GPT2Tokenizer, AutoModelForVision2Seq
import requests

model_name = "mozilla/distilvit"


# Function to load words from a URL
def load_words_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    words = {line.strip() for line in response.text.splitlines()}
    return words


# Load the bad words list
bad_words = load_words_from_url(
    "https://raw.githubusercontent.com/snguyenthanh/better_profanity/master/better_profanity/profanity_wordlist.txt"
)

tokenizer_with_prefix_space = GPT2Tokenizer.from_pretrained(
    model_name, add_prefix_space=True
)


def get_tokens_as_list(word_list):
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space(
            [word], add_special_tokens=False
        ).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


bad_word_ids = get_tokens_as_list(bad_words)

# save the new config on disk
model = AutoModelForVision2Seq.from_pretrained(model_name)
model.generation_config.update(bad_words_ids=bad_word_ids)
model.generation_config.to_json_file("generation_config.json")
