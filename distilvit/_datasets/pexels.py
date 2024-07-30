"""
Tokenizes the Flickr30k dataset
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("pexels")
def get_dataset(feature_extractor_model, text_decoder_model, args):
    from datasets import load_dataset

    split = f"train[:{args.sample}]" if args.sample else "train"
    ds = load_dataset("tarekziade/pexels-gpt4o", split=split)
    # make alt_text a list
    ds = ds.map(lambda ex: {"alt_text": [ex["alt_text"]]})

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="alt_text",
    )

    ds = ds_tokenizer("pexels", ds)

    # splitting in train (80%), test (10%), validation (10%)
    ds = ds.train_test_split(test_size=0.2)
    test_and_eval = ds["test"].train_test_split(test_size=0.5)
    ds["test"] = test_and_eval["test"]
    ds["validation"] = test_and_eval["train"]
    return ds
