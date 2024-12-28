import json
import pandas as pd
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm, trange
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import time

from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dataset_generation.log"), logging.StreamHandler()],
)


class LLMInterface:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
                # torch_dtype=torch.float16, device_map="auto"
            )
            self.model = self.model.to(self.device)

            self.tokenizer.padding_side = "left"

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # Also set the pad token ID in the model config
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            logging.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logging.error(f"Error loading model {self.model_name}: {str(e)}")
            raise

    def generate_response(
        self, prompt_batch: List[str], decoding_strategy
    ) -> Tuple[str, int]:
        try:
            input_lengths = [
                len(self.tokenizer.encode(prompt)) for prompt in prompt_batch
            ]

            print(input_lengths)
            
            max_new_tokens = 400

            inputs = self.tokenizer(
                prompt_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            # print(inputs.input_ids.shape)

            # Generate with standard parameters
            if decoding_strategy == "greedy_search":
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    # temperature=0.7,
                    # do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    length_penalty=0.6,
                    temperature=1.0,
                    early_stopping=True
                )

            elif decoding_strategy == "beam_search":
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    # temperature=0.7,
                    # do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=1.0,
                    early_stopping=True
                )

            # elif decoding_strategy == "constrastive_search":
            #     outputs = self.model.generate(
            #         **inputs,
            #         penalty_alpha=0.6,
            #         top_k=4,
            #         max_new_tokens=max_new_tokens,
            #         temperature=1.0,
            #         early_stopping=True
            #     )

            # elif decoding_strategy == "multinomial_sampling_search":
            #     outputs = self.model.generate(
            #         **inputs,
            #         do_sample=True,
            #         num_beams=1,
            #         max_new_tokens=max_new_tokens,
            #         temperature=1.0,
            #         early_stopping=True
            #     )

            # print(outputs.shape)

            response_batch = []
            output_length_batch = []

            for idx, (output, input_length) in enumerate(zip(outputs, input_lengths)):
                non_pad_start = (
                    (output != self.tokenizer.pad_token_id).nonzero()[0].item()
                )
                input_end = non_pad_start + input_length
                generated_tokens = output[input_end:]

                # breakpoint()
                eos_positions = (
                    generated_tokens == self.tokenizer.eos_token_id
                ).nonzero()
                if len(eos_positions) > 0:
                    output_length = eos_positions[0].item()
                else:
                    output_length = len(generated_tokens)

                output_length_batch.append(output_length)

                decoded_output = self.tokenizer.decode(
                    output[input_end : input_end + output_length],
                    skip_special_tokens=True,
                ).strip()

                response_batch.append(decoded_output)

            # responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # breakpoint()
            # for i in range(len(prompt_batch)):
            #     response = responses[i]
            #     prompt = prompt_batch[i]
            #     # for response in responses:
            #     print(outputs[i])
            #     print(inputs.input_ids[i])
            #     output_length = len(outputs[i]) - len(inputs.input_ids[i])
            #     response = response[len(prompt):].strip()
            #     response_batch.append(response)
            #     output_length_batch.append(output_length)

            print(output_length_batch)

            # breakpoint()

            return response_batch, output_length_batch
            # return response, output_length
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return ["" for _ in range(len(prompt_batch))], [
                0 for _ in range(len(prompt_batch))
            ]


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    prompt = f"Question: {question}\n\nChoices:\n"
    for idx, choice in enumerate(choices):
        prompt += f"{chr(65 + idx)}. {choice}\n"
    prompt += "\nProvide your answer and your explanation in the format 'ANSWER: answer, EXPLANATION: explanation', for example 'ANSWER: A, Explanation: ...'"
    return prompt


def evaluate_correctness(model_response: str, correct_answer: str) -> float:
    # Extract the first letter from the model's response
    """
    Evaluate if the model's response matches the correct answer by looking for the ANSWER: prefix.

    Args:
        model_response: The full response from the model
        correct_answer: The correct answer letter (e.g., 'A', 'B', etc.)
        choices: List of all possible answer choices

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    answer_marker = "ANSWER: "

    try:
        # Find the position of ANSWER: prefix
        marker_pos = model_response.find(answer_marker)
        if marker_pos == -1:
            return 0.0

        # Extract the text after ANSWER: prefix
        answer_text = model_response[marker_pos + len(answer_marker) :].strip()

        # Get the first letter after the prefix
        for char in answer_text:
            if char.isalpha() and char.isupper():
                print(char, correct_answer)
                return 1.0 if char == correct_answer else 0.0

    except Exception as e:
        logging.warning(f"Error evaluating response: {str(e)}")
        return 0.0

    return 0.0


def process_dataset(
    llm_interface: LLMInterface, dataset_samples: List[Dict], batch_size, results_dict
):
    length = len(dataset_samples["question"])
    questions = dataset_samples["question"]
    options = dataset_samples["options"]
    answers = dataset_samples["answer"]
    answer_index = dataset_samples["answer_index"]
    
    decoding_strategies = [
        "greedy_search", 
        "beam_search", 
        # "constrastive_search",
        # "multinomial_sampling_search"
    ]

    total_correctness = 0

    n_batches = length // batch_size

    for decoding_strategy in tqdm(decoding_strategies):
        for i in trange(n_batches):
            print(f"Processing question batch {i+1}/{n_batches}")
            question_batch = questions[i * batch_size : (i + 1) * batch_size]
            option_batch = options[i * batch_size : (i + 1) * batch_size]
            correct_answer_batch = answers[i * batch_size : (i + 1) * batch_size]

            # Format prompt
            prompt_batch = [
                format_mmlu_prompt(question, option)
                for question, option in zip(question_batch, option_batch)
            ]
            # print(prompt)
            # breakpoint()

            # Generate response
            response_batch, output_length_batch = llm_interface.generate_response(
                prompt_batch,
                decoding_strategy=decoding_strategy
            )
            # print(response_batch, output_length_batch, "\n")

            # Evaluate correctness
            correctness_batch = []
            for i in range(len(response_batch)):
                response = response_batch[i]
                correct_answer = correct_answer_batch[i]
                correctness = evaluate_correctness(response, correct_answer)
                correctness_batch.append(correctness)
                total_correctness += correctness

            # Store result
            for i in range(batch_size):
                question_id = str(hash(question_batch[i]))

                # Initialize question entry if it doesn't exist
                if question_id not in results_dict:
                    results_dict[question_id] = {
                        "question": question_batch[i],
                        "option": option_batch[i],
                        "ground_truth": {"answer": correct_answer_batch[i]},
                        "predictions": {},
                    }

                # Add model predictions
                if llm_interface.model_name not in results_dict[question_id]["predictions"]:
                    results_dict[question_id]["predictions"][llm_interface.model_name] = {}
                
                
                results_dict[question_id]["predictions"][llm_interface.model_name][decoding_strategy] = {
                    "predicted_answer": response_batch[i],
                    "answer_length": output_length_batch[i],
                    "correctness": correctness_batch[i],
                }

                # Save intermediate results periodically
                # if len(results_dict) % 100 == 0:
                #     save_results(results_dict, "llm_router_dataset.json")

                # Save intermediate results periodically
                # if len(results_dict) % 100 == 0:
                #     save_results(results_dict, "llm_router_dataset.json")
            # Save intermediate results

    print(f"Total correctness: {total_correctness / length}")

    # Save final results
    # pd.DataFrame(results).to_csv(
    #     output_file, index=False, mode="a", header=not os.path.exists(output_file)
    # )


def save_results(results_dict: Dict[str, Any], output_file: str):
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)


def main():
    # Configuration
    models = [
        # "microsoft/phi-2",  # Using Phi-2 as a substitute for gpt-4-mini
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    # output_file = "llm_router_dataset.csv"

    # Load MMLU-pro dataset
    #
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")

    # print(dataset["test"][:2])
    results_dict = {}

    # breakpoint()
    # Process subset of dataset for each model
    for model_name in models:
        try:
            llm = LLMInterface(model_name)
            llm.load_model()

            print(f"Model successfully loaded: {model_name}")

            # Process dataset
            process_dataset(
                llm_interface=llm,
                dataset_samples=dataset["test"][:1000],
                batch_size=8,
                results_dict=results_dict,
            )

            # Clear GPU memory
            del llm.model
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing model {model_name}: {str(e)}")
            continue

    save_results(results_dict, "llm_router_dataset.json")


if __name__ == "__main__":
    main()
