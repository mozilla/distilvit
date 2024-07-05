import json
import os
import time
from openai import OpenAI
import base64
from datasets import load_dataset, DatasetDict
from io import BytesIO
import sqlite3


api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


DB_PATH = "alt_text_cache.db"


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

DATASET_NAME = "nlphuji/flickr30k"


class AltGenerator:
    def __init__(self):
        self.db = CacheDB()

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
        batch["original_caption"] = list(batch["caption"])
        img_ids = [str(img_id) for img_id in batch["img_id"]]

        # Check cache
        cached_alt_texts = self.db.get(img_ids)
        new_images = [
            image
            for img_id, image in zip(img_ids, batch["image"])
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

        batch["caption"] = caps
        return batch


if __name__ == "__main__":
    generator = AltGenerator()

    split = "test"
    dataset = load_dataset(DATASET_NAME, split=split)

    dataset = dataset.map(
        generator,
        batched=True,
        batch_size=BATCH_SIZE,
    )

    dataset = dataset.rename_column("original_caption", "original_alt_text")
    dataset = dataset.rename_column("caption", "alt_text")
    dataset_dict = DatasetDict({"test": dataset})
    dataset_dict.save_to_disk("./dataset")
    dataset_dict.push_to_hub("mozilla/flickr30k-transformed-captions-gpt4o")
