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

class OPTWithClassificationHeads(torch.nn.Module):
    def __init__(self, model_name="facebook/opt-125m"):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Classification heads
        self.correctness_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
        
        self.length_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 5)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Use [CLS] token representation (first token)
        cls_representation = last_hidden_state[:, 0, :]
        
        correctness_pred = self.correctness_head(cls_representation)
        length_pred = self.length_head(cls_representation)
        
        return correctness_pred.squeeze(), length_pred.squeeze()

class LLMPredictionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.process_data(data)
    
    def process_data(self, data):
        for item in data:
            input_text = f"""Given the question {item['question']} and its option {item['option']}, predict the correctness probability and output length of using the model as {item['model_name']} and the decoding strategy as {item['strategy']} to answer this question."""
            
            encoded = self.tokenizer(
                input_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            self.samples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'correctness': torch.tensor(item['correctness'], dtype=torch.float),
                'length': torch.tensor(item['length'], dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def create_data_splits(
    json_data, tokenizer, train_size=0.7, val_size=0.2, test_size=0.1, random_seed=42
):
    # """
    # Create stratified train, validation, and test splits of the data.
    # """
    # First, organize data by correctness value to ensure stratification
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
                        "length": length // 100,
                    }
                )
    # breakpoint()

    # Create stratified splits

    random.seed(random_seed)

    random.shuffle(total_data)
    
    train_idx = int(len(total_data) * train_size) // 16 * 16
    
    val_idx = int(len(total_data) * (train_size + val_size)) // 16 * 16
    
    train_data = total_data[: train_idx]
    
    print("Correctly answered data num: {}".format(len([d for d in train_data if d["correctness"] == 1])))
    print("Incorrectly answered data num: {}".format(len([d for d in train_data if d["correctness"] == 0])))
    
    print("Output length distribution: {}".format(np.unique([d["length"] for d in train_data], return_counts=True)))
    breakpoint()

    val_data = total_data[
        train_idx : val_idx
    ]

    test_data = total_data[val_idx:]
    
    # breakpoint()

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
    def __init__(self, model_name="facebook/opt-125m", device="cuda:4"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = OPTWithClassificationHeads(model_name).to(self.device)
        
    def train(self, train_dataset, val_dataset, num_epochs=10, train_batch_size=32, eval_batch_size=32, learning_rate=5e-5):
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        # mse_loss = torch.nn.MSELoss()
        ce_loss = torch.nn.CrossEntropyLoss()
        bce_loss = torch.nn.BCELoss()
        
        best_val_loss = float('inf')
        best_correctness_acc = 0
        best_length_acc = 0
        best_model_state = None
        
        for epoch in trange(num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            for idx, batch in enumerate(tqdm(train_dataloader)):
                input_ids = batch['input_ids'].to(self.device)
                # print(input_ids.shape)
                
                attention_mask = batch['attention_mask'].to(self.device)
                correctness = batch['correctness'].to(self.device)
                length = batch['length'].to(self.device)
                
                correctness_pred, length_pred = self.model(input_ids, attention_mask)
                
                # breakpoint()
                loss = bce_loss(correctness_pred, correctness) + ce_loss(length_pred, length)
                total_train_loss += loss.item()
                
                if idx % 50 == 0 and idx > 0:
                    print(f"Loss: {loss.item()}")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            train_loss = total_train_loss / len(train_dataloader)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Metrics: {val_metrics}")
            
            # if val_metrics['total_loss'] < best_val_loss:
            if val_metrics["correctness_accuracy"] > best_correctness_acc and val_metrics["length_accuracy"] > best_length_acc:
                best_correctness_acc = val_metrics["correctness_accuracy"]
                best_length_acc = val_metrics["length_accuracy"]
                best_model_state = self.model.state_dict().copy()
            #     best_val_loss = val_metrics['total_loss']
            #     best_model_state = self.model.state_dict().copy()
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correctness_errors = []
        length_errors = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                correctness = batch['correctness'].to(self.device)
                length = batch['length'].to(self.device)
                
                correctness_pred, length_pred = self.model(input_ids, attention_mask)
                
                # breakpoint()
                length_pred = torch.argmax(length_pred, dim=-1)
                
                # breakpoint()
                
                # loss = bce_loss(correctness_pred, correctness) + mse_loss(length_pred, length)
                # total_loss += loss.item()
                
                correctness_errors.extend(abs(correctness_pred - correctness).cpu().numpy())
                length_errors.extend(abs(length_pred - length).cpu().numpy())
        
        return {
            'correctness_mae': sum(correctness_errors) / len(correctness_errors),
            'length_mae': sum(length_errors) / len(length_errors),
            'correctness_accuracy': sum(e < 0.5 for e in correctness_errors) / len(correctness_errors),
            'length_accuracy': sum(e == 0 for e in length_errors) / len(length_errors)
        }
    
    def predict(self, question, option, model_name, strategy):
        input_text = f"""Given the question {question} and its option {option}, predict the correctness probability and output length of using the following model {model_name} with the strategy {strategy} to answer this question."""
        
        encoded = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            correctness_pred, length_pred = self.model(**encoded)
        
        return {
            'correctness_probability': correctness_pred.item(),
            'predicted_length': length_pred.item()
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
    predictor.train(train_dataset, val_dataset, train_batch_size=16, eval_batch_size=16)

    # Evaluate on test set
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    test_metrics = predictor.evaluate(test_dataloader)
    print("\nTest Set Metrics:")
    # print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Correctness accuracy: {test_metrics['correctness_accuracy']:.4f}, Length accuracy: {test_metrics['length_accuracy']:.4f}")


if __name__ == "__main__":
    main()