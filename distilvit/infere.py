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
image_captioner = pipeline("image-to-text", model="mozilla/distilvit")
git_image_captioner = pipeline("image-to-text", model="microsoft/git-base")


logging.set_verbosity(40)
results = []

for image in IMAGES:
    start = time.time()

    try:
        res1 = git_image_captioner(image, max_new_tokens=20)
    finally:
        duration1 = time.time() - start

    start = time.time()

    try:
        res2 = image_captioner(image, max_new_tokens=20)
    finally:
        duration2 = time.time() - start

    results.append(
        {
            "microsoft/git-base": {
                "time": duration1,
                "result": res1[0]["generated_text"],
            },
            "mozilla/distilvit": {
                "time": duration2,
                "result": res2[0]["generated_text"],
            },
        }
    )


pprint.pprint(results)
