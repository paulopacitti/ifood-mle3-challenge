from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer as TransfomersTrainer,
    TrainingArguments,
)
from datasets import load_from_disk
from evaluate import load
import numpy as np
import nltk
import logging
import argparse

nltk.download("punkt")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


class Trainer:
    def __init__(
        self,
        model_id,
        train_dataset_path,
        test_dataset_path,
        batch_size,
        iterations,
        device="cpu",
    ):
        self.train_dataset = load_from_disk(train_dataset_path).shuffle()
        self.test_dataset = load_from_disk(test_dataset_path).shuffle()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.batch_size = batch_size
        self.iterations = iterations
        self.metric = load("accuracy")

        label2id = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
            "5": 4,
        }
        id2label = {idx: label for label, idx in label2id.items()}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=5, torch_dtype="auto"
        )
        self.model.config.label2id = label2id
        self.model.config.id2label = id2label
        self.model.to(self.device)

    def tokenize(self):
        logging.info("Started tokenizing train dataset.")
        self.train_tokenized_dataset = self.train_dataset.map(
            self._tokenize_function, batched=True
        )
        logging.info("Started tokenizing test dataset.")
        self.test_tokenized_dataset = self.test_dataset.map(
            self._tokenize_function, batched=True
        )
        logging.info("Finished tokenizing datasets.")

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return self.metric.compute(predictions=predictions, references=labels)

    def train(self):
        logging.info("Started training.")
        training_args = TrainingArguments(
            "output",
            eval_strategy="steps",
            eval_steps=self.iterations,
            max_steps=self.iterations,
            per_device_train_batch_size=self.batch_size,
        )

        self.trainer = TransfomersTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_tokenized_dataset,
            eval_dataset=self.test_tokenized_dataset,
            compute_metrics=self.compute_metrics,
        )
        self.trainer.train()
        logging.info("Finished training.")

    def save(self, path):
        logging.info("Saving model.")
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        logging.info("Saved model.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model.")

    parser.add_argument("--model-id", type=str, required=True, help="Model ID")
    parser.add_argument(
        "--train-dataset-path", type=str, required=True, help="Path to train dataset"
    )
    parser.add_argument(
        "--test-dataset-path", type=str, required=True, help="Path to test dataset"
    )
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--iterations", type=int, required=True, help="Iterations")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--save-model-path", type=str, required=True, help="Path to save the model"
    )
    args = parser.parse_args()

    trainer = Trainer(
        model_id=args.model_id,
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        batch_size=args.batch_size,
        iterations=args.iterations,
        device=args.device,
    )
    trainer.tokenize()
    trainer.train()
    trainer.save(args.save_model_path)


if __name__ == "__main__":
    main()
