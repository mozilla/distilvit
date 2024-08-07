"""
Validation dataset used to fine-tune finetuned model
"""
from datasets import DatasetDict
from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("validation")
def get_dataset(feature_extractor_model, text_decoder_model, args):
    from datasets import load_dataset

    ds = load_dataset("Mozilla/alt-text-validation", split="train")

    # keeping only the images with non-empty gpt_alt_text
    ds = ds.filter(lambda x: x["gpt_alt_text"] and x["gpt_alt_text"].strip() != "")

    # copy over the alt_text
    ds = ds.map(lambda x: {"alt_text": [x["gpt_alt_text"]]})

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="alt_text",
    )

    ds = ds_tokenizer("validation", ds)

    # creating a split (we copy over because the ds is very small)
    val_test_size = int(len(ds) * 0.1)
    validation = ds.select(range(val_test_size))
    test = ds.select(range(val_test_size, 2 * val_test_size))
    ds = DatasetDict({"validation": validation, "test": test, "train": ds})
    return ds
