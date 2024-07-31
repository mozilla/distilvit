import base64
import mysql.connector
import json
from PIL import Image
from openai import OpenAI
import os
from io import BytesIO
import time

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


BATCH_SIZE = 5
PROMPT = f"""\
Look at the {BATCH_SIZE} images and create an alternative text and list of detected objects for each.
The produced text is used to train a small image-to-text language model.
You will make it inclusive and eliminate gendered language, racism, sexism, ageism, and ableism.

Guidelines:
- No bias or stereotypes
- Use noun phrases
- The output should be a single sentence of around 10 words.
- No ethnic, racial, or religious markers
- Do not mention text in images
- When an image has a very young person in it , use 'child' or 'kid' instead of "girl" or "boy" to avoid misgendering
- When an image has an adult in it, use "person" instead of "woman" or man" to avoid misgendering
- Use a casual and conversational tone.
- Prefer the word 'person' over 'individual'.
- Use general terms like 'persons,' 'objects,' or 'animals' to describe elements and don't specify their exact number.
- Use indefinite quantifiers such as 'a', 'some', 'many', 'few', or 'all'.
- Avoid "Some people", just say "People"
- Do not count over two, always use indefinite quantifiers over two.
- The text should be understandable by an 8 years old. Use the simplest words possible.
- Try not to lose details of important elements, but keep it as concise as possible.
"""


class OpenaiCache:
    def __init__(
        self,
        host="localhost",
        user="distilvit",
        password="distilvit",
        database="distilvit",
        table="alt_text",
    ):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self._conn = None
        self._cursor = None

    @property
    def cursor(self):
        if self._cursor is not None:
            return self._cursor

        # trigge the propert
        conn = self.conn
        return self._cursor

    @property
    def conn(self):
        if self._conn is not None:
            return self._conn

        self._conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )
        self._cursor = self._conn.cursor()
        self._cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                json_key VARCHAR(255) NOT NULL UNIQUE,
                payload JSON NOT NULL
            )
        """
        )
        self._conn.commit()
        return self._conn

    def _lock_table(self):
        self.cursor.execute(f"LOCK TABLES {self.table} WRITE")

    def _unlock_table(self):
        self.cursor.execute("UNLOCK TABLES")

    def get(self, ids):
        self._lock_table()
        try:
            placeholders = ", ".join(["%s"] * len(ids))
            query = f"SELECT json_key, payload FROM {self.table} WHERE json_key IN ({placeholders})"
            self.cursor.execute(query, ids)
            result = self.cursor.fetchall()
            return {row[0]: json.loads(row[1]) for row in result}
        finally:
            self._unlock_table()

    def set(self, ids, payloads):
        if len(ids) != len(payloads):
            print(ids)
            print(payloads)
            raise ValueError("Length of ids and payloads must match")

        self._lock_table()
        try:
            for i in range(len(ids)):
                query = (
                    f"INSERT INTO {self.table} (json_key, payload) VALUES (%s, %s) "
                    f"ON DUPLICATE KEY UPDATE payload = VALUES(payload)"
                )
                self.cursor.execute(query, (ids[i], json.dumps(payloads[i])))
            self.conn.commit()
        finally:
            self._unlock_table()

    def close(self):
        self.cursor.close()
        self.conn.close()


class AltGenerator:
    def __init__(self, args):
        self.db = OpenaiCache()
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
        if "image" not in batch:
            batch["image"] = [
                Image.open(path).convert("RGB") for path in batch["image_path"]
            ]
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
            generation = self.generate(new_images, new_img_ids)
            if "images" not in generation:
                generation = {"images": [generation]}

            self.db.set(new_img_ids, generation["images"])
            cached_alt_texts.update(dict(zip(new_img_ids, generation["images"])))

        batch[self.args.generated_alt_text_column] = [
            cached_alt_texts[img_id]["alt_text"] for img_id in img_ids
        ]
        batch["objects"] = [
            cached_alt_texts[img_id]["detected_objects"] for img_id in img_ids
        ]
        return batch
