import json
import os
import time
import base64
import argparse
import sqlite3
from io import BytesIO
import pandas as pd

from PIL import Image
from openai import OpenAI
from datasets import load_dataset, DatasetDict, Dataset


api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


DB_PATH = "alt_text_cache.db"


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


class CacheDB:
    def __init__(self, path=DB_PATH):
        self.path = path
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS alt_text_cache (
                img_id TEXT PRIMARY KEY,
                alt_text TEXT,
                objects TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def save(self, img_ids, data):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        for img_id, img_data in zip(img_ids, data):
            c.execute(
                "REPLACE INTO alt_text_cache (img_id, alt_text, objects) VALUES (?, ?, ?)",
                (img_id, img_data["alt_text"], ",".join(img_data["objects"])),
            )
        conn.commit()
        conn.close()

    def get(self, img_ids):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute(
            "SELECT img_id, alt_text, objects FROM alt_text_cache WHERE img_id IN ({})".format(
                ",".join("?" for _ in img_ids)
            ),
            img_ids,
        )
        results = c.fetchall()
        conn.close()
        return {
            img_id: {"alt_text": alt_text, "objects": objects.split(",")}
            for img_id, alt_text, objects in results
        }


BATCH_SIZE = 5
PROMPT = f"""\
Look at the {BATCH_SIZE} images and create an alternative text and list of detected objects for each.
You will make it inclusive and eliminate gendered language, racism, sexism, ageism, and ableism.

Guidelines:
- No bias or stereotypes
- Use noun phrases
- No ethnic, racial, or religious markers
- If there's a girl or boy, use 'child' or 'kid', same for `man` or `woman` so we don't misgender people.
- The output should be a single sentence.
- Use a casual and conversational tone.
- Prefer the word 'person' over 'individual'.
- The text should be understandable by an 8 years old. Use the simplest words possible.
- Try not to lose details of important elements, but keep it as concise as possible.
- The JSON returned is an list of `alt_text` and `objects`
"""


class AltGenerator:
    def __init__(self, args):
        self.db = CacheDB()
        self.args = args

    def image_message(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte)
        encoded_image = img_base64.decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpg;base64,{encoded_image}"},
        }

    def generate(self, images, ids):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            }
        ]

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
            ],
        }

        for image in images:
            user_message["content"].append(self.image_message(image))

        messages.append(user_message)

        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=messages,
                    temperature=0.0,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Failed on attempt {i+1}/3")
                print(images)
                print(ids)
                if i == 2:
                    raise
                time.sleep(1)

    def __call__(self, batch):
        # batch["original_caption"] = list(batch["caption"])
        img_ids = [str(img_id) for img_id in batch[self.args.image_id_column]]
        # Check cache
        cached_alt_texts = self.db.get(img_ids)
        new_images = [
            image
            for img_id, image in zip(img_ids, batch[self.args.image_column])
            if img_id not in cached_alt_texts
        ]
        new_img_ids = [img_id for img_id in img_ids if img_id not in cached_alt_texts]

        if new_images:
            import pdb

            pdb.set_trace()

            generation = self.generate(new_images, new_img_ids)
            if "images" not in generation:
                generation = {"images": [generation]}

            self.db.save(new_img_ids, generation["images"])
            cached_alt_texts.update(dict(zip(new_img_ids, generation["images"])))

        batch[self.args.generated_alt_text_column] = [
            cached_alt_texts[img_id]["alt_text"] for img_id in img_ids
        ]
        batch["objects"] = [cached_alt_texts[img_id]["objects"] for img_id in img_ids]
        return batch


def image_loader(batch, image_column_name="image"):
    images = []
    for image_path in batch["image_path"]:
        try:
            images.append(Image.open(image_path).convert("RGB"))
        except Exception:
            pass
    batch[image_column_name] = images
    del batch["image_path"]
    return batch


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
        dataset = dataset.map(image_loader, batched=True, batch_size=BATCH_SIZE)
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split)

    if args.sample is not None:
        dataset = Dataset.from_dict(dataset[: args.sample])

    dataset = dataset.map(
        generator,
        batched=True,
        batch_size=BATCH_SIZE,
    )

    # dataset = dataset.rename_column("original_caption", "original_alt_text")
    # dataset = dataset.rename_column("caption", "alt_text")

    if args.dataset != "coco":
        dataset_dict = DatasetDict({args.dataset_split: dataset})
    else:
        dataset_dict = dataset

    dataset_dict.save_to_disk("./dataset")
    dataset_dict.push_to_hub(args.target_dataset)
