import logging
import argparse
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def format_dataset(example):
    return {
        "label": example["label"],
        "text": example["text"],
    }


def main(dataset_id, train_test_split_ratio):
    logging.info("Loading dataset...")
    dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)
    dataset = dataset.shuffle().select(range(1000))
    dataset = dataset.train_test_split(float(train_test_split_ratio))
    train_data, test_data = dataset["train"], dataset["test"]

    logging.info("Formating dataset...")
    train_data = train_data.map(format_dataset, remove_columns=train_data.column_names)
    test_data = test_data.map(format_dataset, remove_columns=test_data.column_names)

    logging.info("Saving dataset...")
    train_data.save_to_disk("output/data/train_data")
    test_data.save_to_disk("output/data/test_data")
    logging.info("Data saved to /output. Job done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset.")
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="The hugginface dataset ID to load.",
    )
    parser.add_argument(
        "--train-test-split-ratio",
        type=float,
        required=True,
        help="The train-test split ratio (from 0.01 to 0.99).",
    )

    args = parser.parse_args()
    main(args.dataset_id, args.train_test_split_ratio)
