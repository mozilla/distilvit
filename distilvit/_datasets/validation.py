"""
Validation dataset used to fine-tune finetuned model
"""
from datasets import DatasetDict
from distilvit.utils import DatasetTokenizer, cached_ds


@cached_ds("validation")
def get_dataset(feature_extractor_model, text_decoder_model):
    from datasets import load_dataset

    ds = load_dataset("Mozilla/alt-text-validation", split="train")

    # keeping only the images flagged with need_training
    ds = ds.filter(lambda x: x["need_training"])
    ds = ds.map(lambda x: {"inclusive_alt_text": [x["inclusive_alt_text"]]})

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="inclusive_alt_text",
    )

    ds = ds_tokenizer("validation", ds)

    # creating a split (we copy over because the ds is very small)
    val_test_size = int(len(ds) * 0.1)
    validation = ds.select(range(val_test_size))
    test = ds.select(range(val_test_size, 2 * val_test_size))
    ds = DatasetDict({"validation": validation, "test": test, "train": ds})
    return ds
