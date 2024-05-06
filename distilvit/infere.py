import time
import pprint
from transformers.utils import logging
from transformers import pipeline


IMAGES = [
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg",
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg",
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg",
]

print("Loading model")

CAPTIONERS = [
    ("mozilla/distilvit", pipeline("image-to-text", model="mozilla/distilvit")),
    (
        "microsoft/git-base-coco",
        pipeline("image-to-text", model="microsoft/git-base-coco"),
    ),
    (
        "Salesforce/blip-image-captioning-base",
        pipeline("image-to-text", model="Salesforce/blip-image-captioning-base"),
    ),
]

logging.set_verbosity(40)
results = []

for image in IMAGES:
    for name, image_captioner in CAPTIONERS:
        start = time.time()

        try:
            res = image_captioner(image, max_new_tokens=20)
        finally:
            duration = time.time() - start

        results.append(
            {
                "model": name,
                "time": duration,
                "result": res[0]["generated_text"],
            }
        )


pprint.pprint(results)
