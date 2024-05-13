"""
Tokenizes the Flickr30k dataset
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("flickr30k")
def get_dataset(feature_extractor_model, text_decoder_model):
    from datasets import load_dataset

    ds = load_dataset("nlphuji/flickr30k", split="test")

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="caption",
    )

    ds = ds_tokenizer("flickr30k", ds)

    # splitting in train (80%), test (10%), validation (10%)
    ds = ds.train_test_split(test_size=0.2)
    test_and_eval = ds["test"].train_test_split(test_size=0.5)
    ds["test"] = test_and_eval["test"]
    ds["validation"] = test_and_eval["train"]
    return ds
