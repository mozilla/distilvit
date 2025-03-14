import os
import sys
import shutil

environ_dict = {"NCCL_P2P_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "WANDB_PROJECT": "distilvit",
                "WANDB_LOG_MODEL": "false"
                }

from functools import partial
import torch
from collections.abc import Mapping
import argparse

import nltk
import evaluate
import numpy as np
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import concatenate_datasets, DatasetDict
from transformers.trainer_callback import EarlyStoppingCallback

from distilvit._datasets import DATASETS
from distilvit.quantize import main as quantize
from distilvit.upload import push_to_hub


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)


MAX_LENGTH = 128
THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42
MODEL_ID = "mozilla/distilvit"

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            with open(self.file_path, "a") as f:
                f.write(f"{metrics}\n")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_metrics(
    tokenizer,
    rouge,
    meteor,
    eval_preds,
    args=None,
):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    if args.debug:
        for expected, res in zip(decoded_labels, decoded_preds):
            print(f"Expected: {expected}, got: {res}")

    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["meteor"] = meteor.compute(
        predictions=decoded_preds, references=decoded_labels
    )["meteor"]

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return result


def freeze_model_layers(model, freeze_encoder_layers=3, freeze_decoder_layers=3):
    for i, layer in enumerate(model.encoder.encoder.layer):
        if i < freeze_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

    for i, layer in enumerate(model.decoder.transformer.h):
        if i < freeze_decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False


def data_collator(tokenizer, features):
    # XXX change this so it also works with flickr's labels_0, labels_1 etc
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                # make sure we pad or truncate
                if k == "labels":
                    truncated_features = []
                    for f in features:
                        item = f[k]
                        if len(item) != MAX_LENGTH:
                            print(
                                f"Found item of size {len(item)}), truncating or padding"
                            )
                            if len(item) > MAX_LENGTH:
                                item = item[:MAX_LENGTH]
                            else:
                                item = item + [tokenizer.pad_token_id] * (
                                    MAX_LENGTH - len(item)
                                )

                            assert len(item) == MAX_LENGTH

                        truncated_features.append(item)

                    batch[k] = torch.tensor(truncated_features)
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    return batch


def get_arg_parser(root_dir=None):
    if root_dir is None:
        root_dir = os.path.join(os.path.dirname(__file__), "..")

    parser = argparse.ArgumentParser(
        description="Train a Vision Encoder Decoder Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        type=str,
        help="Model ID",
    )

    parser.add_argument(
        "--sample",
        default=None,
        type=int,
        help="Sample data",
    )

    parser.add_argument(
        "--tag",
        type=str,
        help="HF tag",
        default=None,
    )

    parser.add_argument(
        "--save-dir",
        default=root_dir,
        type=str,
        help="Save dir",
    )

    parser.add_argument(
        "--cache-dir",
        default=os.path.join(root_dir, "cache"),
        type=str,
        help="Cache dir",
    )

    parser.add_argument(
        "--prune-cache",
        default=False,
        action="store_true",
        help="Empty cache dir",
    )

    parser.add_argument(
        "--checkpoints-dir",
        default=os.path.join(root_dir, "checkpoints"),
        type=str,
        help="Checkpoints dir",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode",
    )

    parser.add_argument(
        "--num-train-epochs", type=int, default=3, help="Number of epochs"
    )

    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save steps")

    parser.add_argument(
        "--encoder-model",
        # default="google/vit-base-patch16-224-in21k",
        default="google/vit-base-patch16-224",
        type=str,
        help="Base model for the encoder",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        type=str,
        help="Base model to train again from",
    )
    parser.add_argument(
        "--device",
        default=get_device(),
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Base model to train again from",
    )

    parser.add_argument(
        "--base-model-revision",
        default=None,
        type=str,
        help="Base model revision",
    )

    parser.add_argument("--push-to-hub", action="store_true", help="Push to hub")

    parser.add_argument(
        "--feature-extractor-model",
        default="google/vit-base-patch16-224-in21k",
        #default="google/vit-base-patch16-224",
        type=str,
        help="Feature extractor model for the encoder",
    )
    parser.add_argument(
        "--decoder-model",
        default="distilbert/distilgpt2",
        type=str,
        help="Model for the decoder",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Dataset to use for training",
    )
    return parser

def parse_args(arg_list=None):
    parser = get_arg_parser()
    return parser.parse_args(arg_list)


def train(args):
    get_nltk()
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    feature_extractor = AutoImageProcessor.from_pretrained(args.feature_extractor_model)
    if args.base_model:
        if args.base_model_revision:
            model = VisionEncoderDecoderModel.from_pretrained(
                args.base_model, revision=args.base_model_revision
            )
        else:
            model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

        model_name = f"{args.base_model}+fine-tuned"

    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            args.encoder_model, args.decoder_model
        )

        model_name = (
            f"{args.encoder_model.split('/')[-1]}-{args.decoder_model.split('/')[-1]}"
        )

    #freeze_model_layers(model, freeze_encoder_layers=3, freeze_decoder_layers=3)

    args.device = torch.device(args.device)
    print("Using device", args.device)
    model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
    # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    save_path = os.path.join(args.save_dir, model_name)

    print("Sources", args.dataset)
    datasets = []
    for name in args.dataset:
        get_dataset = DATASETS[name]
        datasets.append(
            get_dataset(
                args.feature_extractor_model,
                args.decoder_model,
                args=args,
            )
        )

    print("Datasets loaded", datasets)
    combined = DatasetDict()
    for split in datasets[0].keys():
        combined[split] = concatenate_datasets([ds[split] for ds in datasets])

    ds = combined.shuffle(seed=THE_ANSWER_TO_LIFE_THE_UNIVERSE_AND_EVERYTHING)

    print("Datasets combined and shuffled", ds)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    training_args = dict(
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=50,
        per_device_eval_batch_size=50,
        num_train_epochs=args.num_train_epochs,
        output_dir=args.checkpoints_dir,
        metric_for_best_model="eval_rougeL",
        save_total_limit=10,
        load_best_model_at_end=True,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        generation_num_beams=2,
        generation_max_length=50
    )

    if args.base_model:
        training_args["generation_config"] = args.model_id

    training_args = Seq2SeqTrainingArguments(**training_args)

    last_checkpoint = get_last_checkpoint(args.checkpoints_dir)
    metrics_logger_callback = MetricsLoggerCallback(
        os.path.join(args.checkpoints_dir, "metrics.txt")
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=partial(compute_metrics,
            tokenizer,
            rouge,
            meteor,
            args=args,
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=partial(data_collator, tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            metrics_logger_callback,
        ],
    )

    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # quantize model
    q_args = [
        "quantize",
        "--model_id",
        save_path,
        "--quantize",
        "--task",
        "image-to-text-with-past",
    ]
    old = sys.argv
    sys.argv = q_args
    try:
        quantize()
    finally:
        sys.argv = old

    print(f"Model saved to {save_path}. You may need to copy in model card in docs directory.")

    if args.push_to_hub:
        push_to_hub(args.model_id, save_path, args.tag, "New training")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
