"""
Using Phi-3-mini-4k-instruct to transform captions from the flickr30k dataset.
"""
import platform
import csv
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset, DatasetDict
from transformers.utils import logging

logging.set_verbosity_error()


class CSVDataSaver:
    def __init__(self, filename="curation.csv"):
        self.filename = filename
        # Ensure the file exists and has the header if it's a new file
        if not Path(self.filename).exists():
            with open(self.filename, mode="w", newline="") as file:
                writer = csv.writer(file, delimiter="|")
                writer.writerow(["dataset", "original caption", "new caption"])

    def add(self, model_id, original_caption, new_caption):
        """Add a new row to the CSV file."""
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerow([model_id, original_caption, new_caption])


csv_saver = CSVDataSaver()


def load_model_and_tokenizer(model_name, device):
    # we have to reduce the memory usage so macbooks don't crash
    if platform.system() == "Darwin":
        kw = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
    else:
        kw = {"trust_remote_code": True}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


# this is the prompt to ask the model to rewrite the caption
# if you find a weird caption, please add an example to this list with the fix.
PROMPT = [
    "Please rewrite the provided caption, focusing on eliminating any form of bias, including racism, sexism, ageism, ableism.",
    "Make it gender-neutral, inclusive, and as close as possible to the original caption with minimal edits.",
    "Prefer the word 'person' over 'individual'.",
    "Prefer 'crowd' over 'diverse group of people'.",
    "Prefer 'kid' over 'child' or 'boy' or 'girl'.",
    "When rewriting 'girl' or 'boy' in the caption, replace it with 'kid'.",
    "Do not add any information that is not present in the original caption, for example 'casual attire'.",
    "Do not change the number of people in the caption. For example, 'two men' cannot be converted to 'a person'.",
    "When the caption mentions 'a dog' or 'a cat' without mentioning a person:, do not replace with 'a person with a dog' or 'a person with a cat'.",
    "Do not generate a caption that is longer than the original caption and don't use a more formal tone.",
    "When you change a word in the caption, make sure it is a synonym of the original word.",
    "Example: 'A man in a blue shirt is standing on a ladder cleaning a window.' becomes 'A person in a blue shirt is standing on a ladder cleaning a window'.",
    "Example 2: 'Three men on a large rig.' becomes: 'A group of people on a large rig.'.",
    "Example 3: 'Two men hiking in the snowy wilderness.' becomes 'Two people hiking in the snowy wilderness.'",
    "Example 4: 'A girl is on rollerskates talking on her cellphone standing in a parking lot.' becomes 'A kid is on rollerblades conversing on a cellphone while standing in a parking lot.'",
]


def transform(captions, model, tokenizer, device):
    transformed_captions = []

    for caption in captions:
        messages = [
            {"role": "user", "content": " ".join(PROMPT) + " : " + caption},
        ]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=50,  # 32
                # do_sample=True
                # num_beams=2,
            )

        result = tokenizer.decode(
            outputs[0][inputs[0].size().numel() :], skip_special_tokens=True
        )

        print(f"{caption} -> {result}")
        csv_saver.add("flickr30k", caption, result)

        transformed_captions.append(result)

    return transformed_captions


def process_batch(batch, model, tokenizer, device):
    batch["original_caption"] = list(batch["caption"])
    batch["caption"] = [
        transform(captions, model, tokenizer, device) for captions in batch["caption"]
    ]
    return batch


def main(test_sample=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (Nvidia GPU).")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    dataset_name = "nlphuji/flickr30k"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    batch_size = 10

    model, tokenizer = load_model_and_tokenizer(model_name, device)
    split = "test[:100]" if test_sample else "test"
    dataset = load_dataset(dataset_name, split=split)

    dataset = dataset.map(
        lambda batch: process_batch(batch, model, tokenizer, device),
        batched=True,
        batch_size=batch_size,
    )

    dataset_dict = DatasetDict({"test": dataset})
    dataset_dict.save_to_disk("./dataset")
    # pushing on my own space for now.
    dataset_dict.push_to_hub("tarekziade/flickr30k-transformed-captions")


if __name__ == "__main__":
    main(test_sample=True)
