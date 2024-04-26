import time
import pprint
from transformers.utils import logging
from transformers import pipeline

logging.set_verbosity(40)


IMAGES = [
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg",
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg",
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg",
]

image_captioner = pipeline("image-to-text", model="tarekziade/distilvit")
original_image_captioner = pipeline(
    "image-to-text", model="nlpconnect/vit-gpt2-image-captioning"
)


results = []

for image in IMAGES:
    start = time.time()

    try:
        res1 = original_image_captioner(image)
    finally:
        duration1 = time.time() - start

    start = time.time()

    try:
        res2 = image_captioner(image)
    finally:
        duration2 = time.time() - start

    results.append(
        {
            "original": {"time": duration1, "result": res1},
            "distilvit": {"time": duration2, "result": res2},
        }
    )


pprint.pprint(results)
