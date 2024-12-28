import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

from tqdm import tqdm, trange


class LLMPredictionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.process_data(data)

    def process_data(self, data):
        for item in data:
            question = item["question"]
            option = item["option"]
            model_name = item["model_name"]
            strategy = item["strategy"]
            correctness = item["correctness"]
            length = item["length"]

            # Create input text in a structured format
            input_text = f"""Given the question {question} and its option {option}, predict the correctness probability and output length of using the following model {model_name} with the strategy {strategy} to answer this question."""

            # Create target text
            target_text = f"Correctness: {correctness:.2f}, Length: {length}"

            # Tokenize
            tokenized_input = self.tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            tokenized_target = self.tokenizer(
                target_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            self.samples.append(
                {
                    "input_ids": tokenized_input["input_ids"].squeeze(),
                    "attention_mask": tokenized_input[
                        "attention_mask"
                    ].squeeze(),
                    "labels": tokenized_target["input_ids"].squeeze(),
                    "target_mask": tokenized_target["attention_mask"].squeeze(),
                    "metadata": {
                        "correctness": correctness,
                        "length": length,
                    },
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_data_splits(
    json_data, tokenizer, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42
):
    # """
    # Create stratified train, validation, and test splits of the data.
    # """
    # First, organize data by correctness value to ensure stratification
    correctness_groups = defaultdict(list)
    total_data = []
    # Process all examples and group by correctness
    for question_id, question_data in json_data.items():
        question = question_data["question"]
        option = question_data["option"]
        for model_name, model_preds in question_data["predictions"].items():
            for strategy, pred_data in model_preds.items():
                correctness = pred_data["correctness"]
                length = pred_data["answer_length"]
                total_data.append(
                    {
                        "question": question,
                        "option": option,
                        "model_name": model_name,
                        "strategy": strategy,
                        "correctness": correctness,
                        "length": length,
                    }
                )
    # breakpoint()

    # Create stratified splits

    random.seed(random_seed)

    random.shuffle(total_data)

    train_data = total_data[: int(len(total_data) * train_size)]

    val_data = total_data[
        int(len(total_data) * train_size) : int(
            len(total_data) * (train_size + val_size)
        )
    ]

    test_data = total_data[int(len(total_data) * (train_size + val_size)) :]

    # for correctness, examples in correctness_groups.items():
    #     random.shuffle(examples)
    #     n_examples = len(examples)

    #     # print(n_examples)
    #     n_train = int(n_examples * train_size)
    #     n_val = int(n_examples * val_size)

    #     # Split examples
    #     train_examples = examples[:n_train]
    #     val_examples = examples[n_train : n_train + n_val]
    #     test_examples = examples[n_train + n_val :]

    #     # Add to respective splits
    #     for ex in train_examples:
    #         if ex["question_id"] not in train_data:
    #             train_data[ex["question_id"]] = json_data[ex["question_id"]]

    #     for ex in val_examples:
    #         if ex["question_id"] not in val_data:
    #             val_data[ex["question_id"]] = json_data[ex["question_id"]]

    #     for ex in test_examples:
    #         if ex["question_id"] not in test_data:
    #             test_data[ex["question_id"]] = json_data[ex["question_id"]]

    # breakpoint()
    # Create datasets
    train_dataset = LLMPredictionDataset(train_data, tokenizer)
    val_dataset = LLMPredictionDataset(val_data, tokenizer)
    test_dataset = LLMPredictionDataset(test_data, tokenizer)

    # breakpoint()

    return train_dataset, val_dataset, test_dataset


class OPTLLMPredictor:
    def __init__(self, model_name="facebook/opt-125m"):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("cuda:4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

    def evaluate(self, dataloader):
        # """
        # Evaluate the model on a dataloader and return metrics
        # """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        correct_lengths = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # outputs = self.model(
                #     input_ids=input_ids, attention_mask=attention_mask, labels=labels
                # )

                # loss = outputs.loss
                # total_loss += loss.item()

                # Generate predictions for accuracy calculation
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # breakpoint()

                pred_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                
                true_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                # Compare predictions with ground truth
                for i in range(len(pred_texts)):
                    # pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
                    # true_text = self.tokenizer.decode(
                    #     labels[i], skip_special_tokens=True
                    # )
                    pred_text = pred_texts[i]
                    true_text = true_texts[i]

                    # print("Pred:", pred_text + "\n")
                    breakpoint()

                    try:
                        pred_correctness = float(
                            pred_text.split("Correctness: ")[-1].split(",")[0]
                        )
                        true_correctness = batch["metadata"]["correctness"][i].item()

                        print(pred_correctness, true_correctness)

                        pred_length = float(
                            pred_text.split("Length: ")[-1].split(",")[0]
                        )
                        true_length = batch["metadata"]["length"][i].item()
                        print(pred_length, true_length)

                        # Consider prediction correct if within 0.1 of true value
                        if abs(pred_correctness - true_correctness) <= 0.1:
                            correct_predictions += 1

                        if abs(pred_length - true_length) <= 10:
                            correct_lengths += 1

                    except:
                        pass
                    
                    total_predictions += 1

        return {
            "loss": total_loss / len(dataloader),
            "correctness_accuracy": (
                correct_predictions / total_predictions if total_predictions > 0 else 0
            ),
            "length_accuracy": (
                correct_lengths / total_predictions if total_predictions > 0 else 0
            ),
        }

    def train(
        self,
        train_dataset,
        val_dataset,
        num_epochs=3,
        train_batch_size=8,
        eval_batch_size=8,
        learning_rate=5e-5,
    ):
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in trange(num_epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            for idx, batch in enumerate(tqdm(train_dataloader)):
                # print(batch["input_ids"].shape)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx % 50 == 0 and idx > 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, Step {idx}/{len(train_dataloader)}: Loss {total_train_loss / (epoch * len(train_dataloader) + idx):.4f}"
                    )

            # Validation phase
            val_metrics = self.evaluate(val_dataloader)
            train_loss = total_train_loss / len(train_dataloader)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            # print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print(
                f"[Validation Accuracy] Correctness: {val_metrics['correctness_accuracy']:.4f}, Length: {val_metrics['length_accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_state = self.model.state_dict().copy()

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, question, option, model_name, decoding_strategy, max_new_tokens=50):
        input_text = f"""Given the question {question} and its option {option}, predict the correctness probability and output length of using the following model: {model_name} with the strategy {decoding_strategy} to answer this question."""
        inputs = self.tokenizer(
            input_text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            correctness_str = predicted_text.split("Correctness: ")[-1].split(",")[0]
            length_str = predicted_text.split("Length: ")[-1].split()[0]

            correctness = float(correctness_str)
            length = int(length_str)

            return {
                "correctness_probability": correctness,
                "predicted_length": length,
                "raw_prediction": predicted_text,
            }
        except:
            return {
                "error": "Failed to parse prediction",
                "raw_prediction": predicted_text,
            }


def main():
    # Load data
    with open("llm_router_dataset.json", "r") as f:
        data = json.load(f)

    # Initialize predictor
    predictor = OPTLLMPredictor()

    # Create data splits
    train_dataset, val_dataset, test_dataset = create_data_splits(
        data, predictor.tokenizer, train_size=0.7, val_size=0.15, test_size=0.15
    )

    print(f"Dataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    # Train the model
    # predictor.train(train_dataset, val_dataset, train_batch_size=32, eval_batch_size=32)

    # Evaluate on test set
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    test_metrics = predictor.evaluate(test_dataloader)
    print("\nTest Set Metrics:")
    # print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Correctness accuracy: {test_metrics['correctness_accuracy']:.4f}, Length accuracy: {test_metrics['length_accuracy']:.4f}")


if __name__ == "__main__":
    main()
