import os
import functools
import re

from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
)
from datasets import Dataset


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
        model_inputs = {}
        model_inputs["labels"] = self.tokenize(examples[self.caption_column])
        model_inputs["pixel_values"] = self.feature_extractor(
            images=examples[self.image_column], return_tensors="pt"
        )["pixel_values"]
        return model_inputs


class DatasetTokenizer:
    def __init__(
        self,
        feature_extractor_model,
        text_decoder_model,
        caption_column="captions",
        image_column="image",
        image_preprocessor_cls=ImagePreprocessor,
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

    def __call__(self, ds):
        ds = ds.map(
            function=self.image_preprocessor,
            batched=True,
            #remove_columns=ds.column_names,
        )
        ds = ds.map(

            lambda example: {
                self.caption_column: to_gender_neutral(example[self.caption_column])
            }
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
    "her": "them",
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


def _replace_match(match):
    return GENDER_DICT[match.group(0)]


def to_gender_neutral(text):
    res = re.sub(GENDER_RE, _replace_match, text)
    if text != res:
        # to log
        print(f"{text} => {res}")
    return res


if __name__ == "__main__":
    text = "A Policeman and a salesman are walking by the road."
    assert (
        to_gender_neutral(text)
        == "A Police Officer and a salesperson are walking by the road."
    )
    print(to_gender_neutral("A woman is holding a boy."))
