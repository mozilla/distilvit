"""
Tokenizes the Docornot DS to recognize scanned documents.
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("docornot")
def get_dataset(feature_extractor_model, text_decoder_model):
    from datasets import load_dataset

    ds = load_dataset("mozilla/docornot", split="train[:2000]")

    # we're only interested in documents and provide a fixed caption.
    ds = ds.filter(lambda example: example["is_document"] == 1)
    ds = ds.map(lambda _: {"caption": ["Text document."]})
    ds = ds.remove_columns("is_document")

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="caption",
    )

    ds = ds_tokenizer("docornot", ds)

    # splitting in train (80%), test (10%), validation (10%)
    ds = ds.train_test_split(test_size=0.2)
    test_and_eval = ds["test"].train_test_split(test_size=0.5)
    ds["test"] = test_and_eval["test"]
    ds["validation"] = test_and_eval["train"]

    return ds
