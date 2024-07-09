import json
import os
import time
import base64
import argparse
import sqlite3
from io import BytesIO

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
                alt_text TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def save(self, img_ids, alt_texts):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        for img_id, alt_text in zip(img_ids, alt_texts):
            c.execute(
                "REPLACE INTO alt_text_cache (img_id, alt_text) VALUES (?, ?)",
                (img_id, alt_text),
            )
        conn.commit()
        conn.close()

    def get(self, img_ids):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute(
            "SELECT img_id, alt_text FROM alt_text_cache WHERE img_id IN ({})".format(
                ",".join("?" for _ in img_ids)
            ),
            img_ids,
        )
        results = c.fetchall()
        conn.close()
        return {img_id: alt_text for img_id, alt_text in results}


BATCH_SIZE = 5
PROMPT = f"""\
Look at the {BATCH_SIZE} images and create an alternative text for each.
You will make it inclusive and eliminate gendered language, racism, sexism, ageism, and ableism.

Guidelines:
- No bias or stereotypes from the text.
- Use noun phrases.
- Remove any ethnic, racial, or religious markers from the text
- If there's a mention of a girl or boy replace it with 'child' or 'kid', same for man or woman so we don't misgender people.
- The output should be a single sentence.
- Use a casual and conversational tone of the text.
- Prefer the word 'person' over 'individual'.
- The text should be understandable by an 8 years old. Use the simplest words possible.
- Try not to lose details of important elements, but keep it as concise as possible.
- The JSON key to use is "alt_text" and is the list of alt texts.
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

                return json.loads(response.choices[0].message.content)["alt_text"]
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
            new_alt_texts = self.generate(new_images, new_img_ids)
            self.db.save(new_img_ids, new_alt_texts)
            cached_alt_texts.update(dict(zip(new_img_ids, new_alt_texts)))

        caps = []
        for img_id in img_ids:
            caps.append(cached_alt_texts[img_id])

        batch[self.args.generated_alt_text_column] = caps
        return batch


if __name__ == "__main__":
    args = parse_args()
    generator = AltGenerator(args)
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
    dataset_dict = DatasetDict({args.dataset_split: dataset})
    dataset_dict.save_to_disk("./dataset")
    dataset_dict.push_to_hub(args.target_dataset)
