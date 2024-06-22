import argparse
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a model to the HF hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-id",
        type=str,
        help="Model ID",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        help="Save dir",
    )

    parser.add_argument(
        "--tag",
        type=str,
        help="HF tag",
        default=None,
    )

    parser.add_argument(
        "--commit-message",
        default="New Training",
        type=str,
        help="Save dir",
    )

    return parser.parse_args()


def push_to_hub(model_id, save_path, tag, commit_message):
    api = HfApi()

    api.upload_folder(
        repo_id=model_id, folder_path=save_path, commit_message=commit_message
    )

    if args.tag:
        api.create_tag(
            repo_id=model_id,
            tag=tag,
        )


if __name__ == "__main__":
    args = parse_args()
    push_to_hub(args.model_id, args.save_path, args.tag, args.commit_message)
