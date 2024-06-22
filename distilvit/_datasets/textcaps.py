"""
Tokenizes the TextCaps dataset
"""

from datasets import load_dataset
from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("textcaps")
def get_dataset(feature_extractor_model, text_decoder_model, args):
    ds = load_dataset("lmms-lab/TextCaps", split="train")

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="caption_str",
        image_column="image",
    )

    ds = ds_tokenizer(ds)

    # splitting in train (80%), test (10%), validation (10%)
    ds = ds.train_test_split(test_size=0.2)
    test_and_eval = ds["test"].train_test_split(test_size=0.5)
    ds["test"] = test_and_eval["test"]
    ds["validation"] = test_and_eval["train"]
    return ds
