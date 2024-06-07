"""
Using Llama 3 8B Instructto transform captions from the flickr30k dataset.
"""
import re
import platform
import torch
import argparse
from transformers.utils import logging
import readability

logging.set_verbosity_error()

DATASET_NAME = "nlphuji/flickr30k"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = platform.system() == "Darwin" and 1 or 10


PROMPT_1 = """
Rewrite the text to be inclusive and free of bias:
- Remove gendered pronouns and names, but not for animals.
- Remove ethnic, racial, and religious markers.
- Maintain the order and relationship of descriptive elements without changing verbs.
- Keep the sentence structure as close to the original as possible.
- Wrap the result in triple backticks.
"""

PROMPT_2 = """
Rewrite the text to:
- Maintain original verbs for a casual tone.
- Use singular forms when the original text describes one person.
- Keep the sentence structure as close to the original as possible.
- Wrap the result in triple backticks.
"""

PROMPT_3 = """
Rewrite the text to use noun phrases for brevity and simplicity:
- Convert sentences to noun phrases where possible: 'a person is walking' becomes 'a person walking'.
- Maintain the order and relationship of descriptive elements without changing verbs.
- Avoid adding new verbs or altering the original ones.
- Match the original sentence length.
- Wrap the result in triple backticks.
"""

PROMPT_4 = """
Rewrite the text to:
- Avoid adding new verbs or altering the original ones.
- Wrap the result in triple backticks.
"""

PROMPTS = [PROMPT_1, PROMPT_3]


class TextConverter:
    def __init__(self, args, model_name=MODEL_NAME):
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
        self.args = args

    def load_model_and_tokenizer(self):
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

    def by_grade(self, item):
        return item[1]["DaleChallIndex"]

    def extract_text_with_backticks(self, input_string):
        pattern = r"```(.*?)```"
        match = re.search(pattern, input_string, re.DOTALL)
        if match is None:
            return input_string
        res = match.group(1).strip()
        if self.args.debug:
            print(f"original:\n{input_string}\nbacktick extracted:\n{res}\n")
        return res

    def transform(self, captions, sentids):
        transformed_captions = []

        for caption, sentid in zip(captions, sentids):
            result = self.transform_one(caption)
            try:
                grade = dict(
                    readability.getmeasures(result, lang="en")["readability grades"]
                )
            except Exception as e:
                grade = {"DaleChallIndex": 10.0}

            print(f"{caption} -> {result} with {grade['DaleChallIndex']:.2f}")
            transformed_captions.append((result, grade, sentid))

        transformed_captions.sort(key=self.by_grade)
        return list(zip(*transformed_captions))

    def transform_one(self, caption):
        if self.model is None:
            self.load_model_and_tokenizer()

        for i, prompt in enumerate(PROMPTS):
            try:
                messages = [
                    {"role": "user", "content": prompt + caption},
                ]

                inputs = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=120,
                        no_repeat_ngram_size=2,
                        repetition_penalty=1.2,
                        num_beams=3,
                        early_stopping=True,
                    )

                result = self.tokenizer.decode(
                    outputs[0][inputs[0].size().numel() :], skip_special_tokens=True
                )
                result = self.extract_text_with_backticks(result)
                result = result.split("\n")[0].strip()
                if self.args.debug:
                    print(f"step {i}: {caption} -> {result}")
                caption = result
            except Exception as e:
                print(f"Failed to process {caption}: {e}")
                return caption

        return caption


def main(args):
    llm_converter = TextConverter(args)

    if args.text:
        result = llm_converter.transform_one(args.text)
        print(f"Transformed Text: {result}")
    else:
        from datasets import load_dataset, DatasetDict

        split = "test[:100]" if args.test_sample else "test"
        dataset = load_dataset(DATASET_NAME, split=split)

        num_proc = platform.system() == "Darwin" and 1 or 4

        dataset = dataset.map(
            llm_converter.process_batch,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=num_proc,
        )
        dataset = dataset.rename_column("original_caption", "original_alt_text")
        dataset = dataset.rename_column("caption", "alt_text")
        dataset_dict = DatasetDict({"test": dataset})
        dataset_dict.save_to_disk("./dataset")
        if not args.test_sample:
            dataset_dict.push_to_hub("mozilla/flickr30k-transformed-captions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some text.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--text", type=str, help="Text to transform")
    parser.add_argument("--test_sample", action="store_true", help="Run a test sample")
    args = parser.parse_args()
    main(args)
