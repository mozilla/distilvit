import datasets
from transformers.utils import logging
from transformers import pipeline
import argparse


def main(before, after):
    # Load the dataset
    dataset = datasets.load_dataset("mozilla/alt-text-validation")

    # Filter images based on need_training
    filtered_images = [
        (item["image"], item["inclusive_alt_text"])
        for item in dataset["train"]
        if item["need_training"]
    ]

    IMAGES = filtered_images

    CAPTIONERS = [
        (
            "before",
            pipeline(
                "image-to-text",
                model=before,
                revision="main",
            ),
        ),
        (
            "after",
            pipeline(
                "image-to-text",
                model=after,
                revision="main",
            ),
        ),
    ]

    logging.set_verbosity(40)

    for image, inclusive_alt_text in IMAGES:
        line = [f"expected: {inclusive_alt_text}"]

        for name, image_captioner in CAPTIONERS:
            res = image_captioner(image, max_new_tokens=40)
            line.append(f"{name}: {res[0]['generated_text']}")

        print(" | ".join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image captioning models")
    parser.add_argument(
        "--before",
        type=str,
        help="Path or name of the first model",
        default="mozilla/distilvit",
    )
    parser.add_argument(
        "--after",
        type=str,
        help="Path or name of the second model",
        default="/Users/tarekziade/Dev/distilvit/distilvit/../vit-base-patch16-224-distilgpt2",
    )
    args = parser.parse_args()

    main(args.before, args.after)
