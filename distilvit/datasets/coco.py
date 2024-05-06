"""
Downloads COCO and creates a tokenized one with extracted features.
"""

import requests
import os

import nltk
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from distilvit.utils import DatasetTokenizer, cached_ds, ImagePreprocessor

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


def download_file(url, directory):
    local_filename = url.split("/")[-1]
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    path_to_file = os.path.join(directory, local_filename)

    # Only download if the file does not exist
    if not os.path.exists(path_to_file):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            with open(path_to_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        return path_to_file
    else:
        print(f"{local_filename} already exists. Skipping download.")
        return path_to_file


urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/image_info_test2017.zip",
]

COCO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "coco")


class CocoImagePreprocessor(ImagePreprocessor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        caption_column="caption",
        image_column="image_path",
    ):
        self.base_feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.caption_column = caption_column
        self.image_column = image_column

    def feature_extractor(self, *args, **kw):
        images = []
        for image_path in kw['images']:
            try:
                images.append(Image.open(image_path).convert("RGB"))
            except Exception:
                pass

        inputs = self.base_feature_extractor(images=images, return_tensors="pt")
        for image in images:
            image.close()
        return inputs


@cached_ds("coco")
def get_dataset(feature_extractor_model, text_decoder_model):
    """Downloads the COCO dataset and tokenizes it.

    The result is saved on disk so we can reuse it.
    """

    for url in urls:
        print(f"Downloading {url}...")
        download_file(url, COCO_DIR)
    print("Download complete.")

    ds = load_dataset(
        "ydshieh/coco_dataset_script",
        "2017",
        data_dir=COCO_DIR,
        trust_remote_code=True,
    )

    ds_tokenizer = DatasetTokenizer(
        feature_extractor_model,
        text_decoder_model,
        caption_column="caption",
        image_column="image_path",
        image_preprocessor_cls=CocoImagePreprocessor,
    )

    ds = ds_tokenizer(ds)
    return ds
