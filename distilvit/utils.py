import os
import functools

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
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
        self.feature_extractor = AutoImageProcessor.from_pretrained(
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

    def logger(self, msg):
        self.logfile.write(msg + "\n")

    def __call__(self, ds_name, ds):
        self.logger(f"Processing {ds_name}")
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
            if len(args) == 3:
                cache_dir = args[-1]
                args = args[:-1]
            else:
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
