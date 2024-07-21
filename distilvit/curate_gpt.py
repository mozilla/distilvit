import argparse
import pandas as pd

from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from distilvit.gpt4 import AltGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate alternative text for images in a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run on a sample",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./dummy_data",
        help="Dataset dir",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="nlphuji/flickr30k",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        default="mozilla/flickr30k-transformed-captions-gpt4o",
        help="Name of the target dataset to save to",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Name of the dataset split",
    )

    parser.add_argument(
        "--image-column",
        type=str,
        default="image",
        help="Name of the image column in the dataset",
    )
    parser.add_argument(
        "--generated-alt-text-column",
        type=str,
        default="alt_text",
        help="Name of the resulting alt text column",
    )
    parser.add_argument(
        "--image-id-column",
        type=str,
        default="img_id",
        help="Name of the image id column in the dataset",
    )

    return parser.parse_args()


def drop_duplicates_in_split(split):
    df = pd.DataFrame(split)
    df_selected = (
        df[["image_id", "image_path", "coco_url"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return Dataset.from_pandas(df_selected)


if __name__ == "__main__":
    args = parse_args()
    generator = AltGenerator(args)

    if args.dataset == "coco":
        try:
            dataset = load_from_disk("./dataset-temp-coco")
        except Exception:
            dataset = load_dataset(
                "ydshieh/coco_dataset_script", "2017", data_dir=args.data_dir
            )

            train_unique = drop_duplicates_in_split(dataset["train"])
            test_unique = drop_duplicates_in_split(dataset["test"])
            validation_unique = drop_duplicates_in_split(dataset["validation"])

            dataset = DatasetDict(
                {
                    "train": train_unique,
                    "test": test_unique,
                    "validation": validation_unique,
                }
            )
            dataset.save_to_disk("./dataset-temp-coco")
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split)

    if args.sample is not None:
        dataset = Dataset.from_dict(dataset[: args.sample])

    dataset = dataset.map(generator, batched=True, batch_size=5, num_proc=4)

    if args.dataset != "coco":
        dataset_dict = DatasetDict({args.dataset_split: dataset})
    else:
        dataset_dict = dataset

    dataset_dict.save_to_disk("./dataset")
    dataset_dict.push_to_hub(args.target_dataset)
