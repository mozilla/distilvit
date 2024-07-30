import os
import requests
from datasets import Dataset, Features, Value, Image as DImage, load_from_disk, Sequence
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd
import concurrent.futures

from distilvit.gpt4 import AltGenerator, OpenaiCache

# Define the Pexels API key and endpoint
PEXELS_API_KEY = os.environ["PEXELS_API_KEY"]
PEXELS_API_URL = "https://api.pexels.com/v1/search?query="

per_page = 80  # Max per page limit for Pexels API
total_images = 100  # Number of images to fetch


def resize_image(img):
    max_size = 700
    width, height = img.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    return img


def fetch_images_from_pexels(api_key, per_page, query, total_images=1000, start_page=1):
    headers = {"Authorization": api_key}
    page = start_page
    num = 0
    while num < total_images:
        response = requests.get(
            PEXELS_API_URL + query,
            headers=headers,
            params={"per_page": per_page, "page": page},
        )
        response_json = response.json()

        if not response_json["photos"]:
            break  # Exit if no more photos are available

        for image in response_json["photos"]:
            yield image
            num += 1

        page += 1


def process_images(image_ids, generator, image_by_id, pbar):
    cache = OpenaiCache()
    cached_images = cache.get(image_ids)
    new_images = []
    new_images_ids = []

    for image_id in image_ids:
        image = Image.open(os.path.join("images", image_id)).convert("RGB")
        byte_arr = BytesIO()
        image.save(byte_arr, format="JPEG")
        image_by_id[image_id] = byte_arr.getvalue()

        if image_id not in cached_images:
            new_images.append(image)
            new_images_ids.append(image_id)

    if new_images_ids:
        generated = generator.generate(new_images, new_images_ids)
        generated = generated["images"]
        cache.set(new_images_ids, generated)

        for id, payload in zip(new_images_ids, generated):
            cached_images[id] = payload

    pbar.update(5)
    return cached_images


def generate():
    entries = []
    generator = AltGenerator({})
    pbar = tqdm(total=5000)

    image_files = [image for image in os.listdir("images") if image.endswith(".jpg")]

    image_by_id = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(0, len(image_files), 5):
            image_ids = image_files[i : i + 5]
            futures.append(
                executor.submit(process_images, image_ids, generator, image_by_id, pbar)
            )

        idx = 0
        for future in concurrent.futures.as_completed(futures):
            cached_images = future.result()
            for image_id, payload in cached_images.items():
                new_entry = {
                    "image_id": idx,
                    "alt_text": payload["alt_text"],
                    "objects": payload["objects"],
                    "image": image_by_id[image_id],
                    "source": "pexels",
                }
                entries.append(new_entry)
                idx += 1

    pbar.close()

    features = Features(
        {
            "image_id": Value("int32"),
            "alt_text": Value("string"),
            "objects": Sequence(Value("string")),
            "image": DImage(),
            "source": Value("string"),
        }
    )

    new_entries_df = pd.DataFrame(entries)
    new_dataset = Dataset.from_pandas(new_entries_df, features=features)
    new_dataset = new_dataset.shuffle(seed=42)
    new_dataset.save_to_disk("new_pexels_dataset")
    new_dataset.push_to_hub("tarekziade/pexels-gpt4o")


generate()

from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_images(query):
    index = 1
    for image in tqdm(
        fetch_images_from_pexels(PEXELS_API_KEY, per_page, query, start_page=1),
        total=1000,
        desc=f"Fetching images from Pexels for {query}",
    ):
        image_url = image["src"]["original"]
        response = requests.get(image_url)
        pil_image = resize_image(Image.open(BytesIO(response.content)).convert("RGB"))
        pil_image.save(os.path.join("images", f"{query}-{index}.jpg"), format="JPEG")
        index += 1


def fetch_pexels():
    queries = ["animals", "nudes", "smoking", "group", "war"]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_images, query) for query in queries]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
