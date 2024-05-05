"""
Tokenizes the Docornot DS to recognize scanned documents.
"""

from datasets import load_dataset
from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("docornot")
def get_dataset(feature_extractor_model, text_decoder_model):
    ds = load_dataset("mozilla/docornot")

    # we are only taking documents and set the caption to
    # "The image seems to be a textual document."
    # XXX

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="caption",
    )

    ds = ds_tokenizer(ds)

    return ds
