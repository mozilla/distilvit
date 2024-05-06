import os
import functools
import re

from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    pipeline,
)


MAX_LENGTH = 128


class ImagePreprocessor:
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        caption_column="captions",
        image_column="image",
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.caption_column = caption_column
        self.image_column = image_column

    def tokenize(self, captions):
        return self.tokenizer(
            captions, padding="max_length", max_length=MAX_LENGTH
        ).input_ids

    def __call__(self, examples):
        if isinstance(examples[self.caption_column], list):
            captions = [cap[0] for cap in examples[self.caption_column]]
        else:
            captions = examples[self.caption_column]

        return {
            "labels": self.tokenize(captions),
            "pixel_values": self.feature_extractor(
                images=examples[self.image_column], return_tensors="pt"
            )["pixel_values"],
        }


class DatasetTokenizer:
    def __init__(
        self,
        feature_extractor_model,
        text_decoder_model,
        caption_column="captions",
        image_column="image",
        image_preprocessor_cls=ImagePreprocessor,
        logfile="replacements.csv",
    ):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.caption_column = caption_column
        self.image_column = image_column
        self.image_preprocessor = image_preprocessor_cls(
            self.feature_extractor,
            self.tokenizer,
            caption_column=self.caption_column,
            image_column=self.image_column,
        )
        self.logfile = open(logfile, "a+")
        self.offensivelog = open("offensive.csv", "a+")
        self.offensive_classifier = pipeline("sentiment-analysis", model="mozilla/pardonmyai")

    def is_offensive(self, text):
        res = self.offensive_classifier(text)
        if res[0]["label"] == "OFFENSIVE":
            self.offensivelog.write(f"{text}\n")
            return True
        return False

    def logger(self, msg):
        self.logfile.write(msg + "\n")

    def cleanup(self, ds_name, batch):
        batch[self.caption_column] = [
            cleanup(ds_name, label, self.logger) for label in batch[self.caption_column]
        ]
        return batch

    def __call__(self, ds_name, ds):
        ds = ds.filter(lambda x: not self.is_offensive(x[self.caption_column])
        ds = ds.map(
            functools.partial(self.cleanup, ds_name),
            batched=True,
        )

        ds = ds.map(
            function=self.image_preprocessor,
            batched=True,
            # remove_columns=ds.column_names,
        )
        return ds


def cached_ds(cache_name):
    def _cached_ds(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_dir = kwargs.pop("cache_dir", ".cache")
            cached_ds = os.path.join(cache_dir, cache_name)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            elif os.path.exists(cached_ds):
                from datasets import load_from_disk

                return load_from_disk(cached_ds)

            ds = func(*args, **kwargs)
            ds.save_to_disk(cached_ds)
            return ds

        return wrapper

    return _cached_ds


GENDER_DICT = {
    # Singular pronouns
    "he": "they",
    "He": "They",
    "she": "they",
    "She": "They",
    "him": "them",
    "Him": "Them",
    "her": "their",
    "Her": "Them",
    "his": "their",
    "His": "Their",
    "hers": "theirs",
    "Hers": "Theirs",
    "himself": "themself",
    "Himself": "Themself",
    "herself": "themself",
    "Herself": "Themself",
    # Plural pronouns
    "guys": "folks",
    "Guys": "Folks",
    "ladies": "folks",
    "Ladies": "Folks",
    "gentlemen": "folks",
    "Gentlemen": "Folks",
    # Specific roles and occupations
    "female": "person",  # XXX not sure about this one
    "male": "person",  # XXX not sure about this one
    "Female": "Person",  # XXX not sure about this one
    "Male": "Person",  # XXX not sure about this one
    "man": "person",
    "Man": "Person",
    "woman": "person",
    "Woman": "Person",
    "men": "people",
    "Men": "People",
    "women": "people",
    "Women": "People",
    "boy": "child",
    "Boy": "Child",
    "girl": "child",
    "Girl": "Child",
    "boys": "children",
    "Boys": "Children",
    "girls": "children",
    "Girls": "Children",
    "father": "parent",
    "Father": "Parent",
    "mother": "parent",
    "Mother": "Parent",
    "grandfather": "grandparent",
    "Grandfather": "Grandparent",
    "grandmother": "grandparent",
    "Grandmother": "Grandparent",
    "son": "child",
    "Son": "Child",
    "daughter": "child",
    "Daughter": "Child",
    "husband": "spouse",
    "Husband": "Spouse",
    "wife": "spouse",
    "Wife": "Spouse",
    "boyfriend": "partner",
    "Boyfriend": "Partner",
    "girlfriend": "partner",
    "Girlfriend": "Partner",
    "king": "ruler",
    "King": "Ruler",
    "queen": "ruler",
    "Queen": "Ruler",
    "actor": "performer",
    "Actor": "Performer",
    "actress": "performer",
    "Actress": "Performer",
    "waiter": "server",
    "Waiter": "Server",
    "waitress": "server",
    "Waitress": "Server",
    "steward": "flight attendant",
    "Steward": "Flight Attendant",
    "stewardess": "flight attendant",
    "Stewardess": "Flight Attendant",
    "chairman": "chairperson",
    "Chairman": "Chairperson",
    "chairwoman": "chairperson",
    "Chairwoman": "Chairperson",
    "policeman": "police officer",
    "Policeman": "Police Officer",
    "policewoman": "police officer",
    "Policewoman": "Police Officer",
    "fireman": "firefighter",
    "Fireman": "Firefighter",
    "firewoman": "firefighter",
    "Firewoman": "Firefighter",
    "salesman": "salesperson",
    "Salesman": "Salesperson",
    "saleswoman": "salesperson",
    "Saleswoman": "Salesperson",
    "mailman": "mail carrier",
    "Mailman": "Mail Carrier",
    "mailwoman": "mail carrier",
    "Mailwoman": "Mail Carrier",
    "milkman": "milk delivery person",
    "Milkman": "Milk Delivery Person",
    "businessman": "businessperson",
    "Businessman": "Businessperson",
    "businesswoman": "businessperson",
    "Businesswoman": "Businessperson",
}

GENDER_RE = r"\b(" + "|".join(re.escape(key) for key in GENDER_DICT.keys()) + r")\b"

NUMBERS_DICT = {
    "two": "some",
    "three": "some",
    "four": "some",
    "five": "some",
    "six": "many",
    "seven": "many",
    "eight": "many",
    "nine": "many",
    "ten": "many",
    "Two": "Some",
    "Three": "Some",
    "Four": "Some",
    "Five": "Some",
    "Six": "Many",
    "Seven": "Many",
    "Eight": "Many",
    "Nine": "Many",
    "Ten": "Many",
}

NUMBERS_RE = r"\b(" + "|".join(re.escape(key) for key in NUMBERS_DICT.keys()) + r")\b"

# When we get a combo of replacement words, we replace it with people
# A man and a woman => a person and a person => people
COMBINED_DICT = {}
for replacement in GENDER_DICT.values():
    if replacement == "":
        continue
    if replacement[0].isupper():
        continue
    COMBINED_DICT[f"a {replacement} and a {replacement}"] = "people"
    COMBINED_DICT[f"A {replacement} and a {replacement}"] = "People"

COMBINED_RE = r"\b(" + "|".join(re.escape(key) for key in COMBINED_DICT.keys()) + r")\b"


def cleanup(ds_name, text, logger=None):
    # for now we drop extra captions
    if isinstance(text, list):
        text = text[0]

    def _replace_match(dikt, match):
        return dikt[match.group(0)]

    # order matters.
    for regexp, dikt in [
        (GENDER_RE, GENDER_DICT),
        (NUMBERS_RE, NUMBERS_DICT),
        (COMBINED_RE, COMBINED_DICT),
    ]:
        res = re.sub(regexp, functools.partial(_replace_match, dikt), text)
        if text != res:
            if logger is not None:
                logger(f"{ds_name}|{text}|{res}")
            text = res

    text = text.strip()
    if text[0].islower():
        text = text[0].upper() + text[1:]

    if not text.endswith("."):
        text += "."

    return text


if __name__ == "__main__":
    for text in [
        "Four Policemen and a salesman are walking by the road.",
        "A woman is holding a boy.",
        "A man and a woman are sitting on a bench.",
    ]:
        res = cleanup("test", text)
        print(f"{text} => {res}")
