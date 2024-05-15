"""
Using Llama 3 8B Instructto transform captions from the flickr30k dataset.
"""
import re
import platform

import torch
from transformers.utils import logging
import readability


logging.set_verbosity_error()


DATASET_NAME = "nlphuji/flickr30k"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 10


def extract_text_with_backticks(input_string):
    pattern = r"```(.*?)```"
    match = re.search(pattern, input_string, re.DOTALL)
    if match is None:
        return input_string
    return match.group(1).strip()


PROMPT = """\
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


class TextConverter:
    def __init__(self, model_name=MODEL_NAME):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA (Nvidia GPU).")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU).")
        else:
            device = torch.device("cpu")
            print("Using CPU.")

        self.device = device
        self.model = None
        self.model_name = model_name

    def load_model_and_tokenizer(self):
        # we have to reduce the memory usage so macbooks don't crash
        if platform.system() == "Darwin":
            kw = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            bnb_config = None
        else:
            from transformers import BitsAndBytesConfig

            kw = {
                "trust_remote_code": True,
            }

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=bnb_config, **kw
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def process_batch(self, batch):
        if self.model is None:
            self.load_model_and_tokenizer()

        # need to re-triage the original captions with the new order
        batch["original_caption"] = list(batch["caption"])
        batch["original_sentids"] = list(batch["sentids"])

        new_captions = []
        grades = []
        sentids = []

        for captions, nsentids in zip(batch["caption"], batch["sentids"]):
            converted, grade, nsentids = self.transform(captions, nsentids)
            new_captions.append(converted)
            grades.append(grade)
            sentids.append(nsentids)

        batch["caption"] = new_captions
        batch["grade"] = grades
        batch["sentids"] = sentids

        return batch

    def transform(self, captions, sentids):
        transformed_captions = []

        for caption, sentid in zip(captions, sentids):
            messages = [
                {"role": "user", "content": PROMPT + caption},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=40,
                )

            result = self.tokenizer.decode(
                outputs[0][inputs[0].size().numel() :], skip_special_tokens=True
            )
            result = extract_text_with_backticks(result)
            result = result.split("\n")[0].strip()
            grade = dict(
                readability.getmeasures(result, lang="en")["readability grades"]
            )

            transformed_captions.append((result, grade, sentid))

        def by_grade(item):
            return item[1]["DaleChallIndex"]

        transformed_captions.sort(key=by_grade)

        return list(zip(*transformed_captions))


def main(test_sample=False):
    from datasets import load_dataset, DatasetDict

    split = "test[:10]" if test_sample else "test"
    dataset = load_dataset(DATASET_NAME, split=split)

    llm_converter = TextConverter()

    num_proc = platform.system() == "Darwin" and 1 or 4

    dataset = dataset.map(
        llm_converter.process_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=num_proc,
    )

    dataset_dict = DatasetDict({"test": dataset})
    dataset_dict.save_to_disk("./dataset")
    # pushing on my own space for now.
    dataset_dict.push_to_hub("tarekziade/flickr30k-transformed-captions")


if __name__ == "__main__":
    main(test_sample=True)
