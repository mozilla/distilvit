"""
Tokenizes the Docornot DS to recognize scanned documents.
"""

from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("docornot")
def get_dataset(feature_extractor_model, text_decoder_model):
    from datasets import load_dataset

    ds = load_dataset("mozilla/docornot")

    # we're only interested in documents and provide a fixed caption.
    ds = ds.filter(lambda example: example["is_document"] == 1)
    ds = ds.map(lambda _: {"caption": ["The image seems to be a textual document."]})
    ds = ds.remove_columns("is_document")

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="caption",
    )

    ds = ds_tokenizer("docornot", ds)

    return ds
