"""
Using Phi-3-mini-4k-instruct to transform captions from the flickr30k dataset.
"""
import re
import platform
import csv
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset, DatasetDict
from transformers.utils import logging

logging.set_verbosity_error()


def extract_text_with_backticks(input_string):
    pattern = r"```(.*?)```"
    match = re.search(pattern, input_string, re.DOTALL)
    if match is None:
        return input_string
    return match.group(1).strip()


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


PROMPT3 = """\
Please rewrite the provided text to make it inclusive and eliminate gendered language, racism, sexism, ageism, and ableism:
- Remove any bias or stereotypes from the text.
- Keep animal descriptions intact. For example, 'a black dog' should remain 'a black dog' and not 'a dog'.
- Remove any ethnic, racial, or religious markers from the text.
- Avoid changing original verbs to maintain the casual and conversational tone of the text.
- Use plural forms or collective nouns to keep the language fluid and natural. For example, instead of saying 'a black woman and a white man,' refer to them as 'two people' or 'workers' if they are performing a task.
- The goal is to maintain a natural flow and avoid awkward repetitions while ensuring the description remains clear and true to the original content.
- Prefer the word `person` over `individual`.
- Do not make count mistakes. For example, if the original text says 'a little girl', replace it with 'a kid'.
- Do not try to describe the scene; focus on just rewriting the text as instructed.
- The output should be a single sentence and its length should be close to the original text.
- The text should be understandable by an 8 years old. Use the simplest words possible.
Wrap the result between triple backticks.

"""


def transform(captions, model, tokenizer, device):
    transformed_captions = []

    for caption in captions:
        messages = [
            {"role": "user", "content": PROMPT3 + caption},
        ]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=40,  # 32
                # do_sample=True,
                # num_beams=2,
            )

        result = tokenizer.decode(
            outputs[0][inputs[0].size().numel() :], skip_special_tokens=True
        )
        result = extract_text_with_backticks(result)
        result = result.split("\n")[0].strip()
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
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # model_name = "microsoft/Phi-3-mini-4k-instruct"
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
