from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class OpenSourceLLM:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.device = device

def create_llm_cores(model_configs):
    """
    Create multiple LLM cores using different open-source models
    
    Args:
        model_configs (list): List of dictionaries containing model configurations
            Each dict should have:
            - 'name': Model identifier from HuggingFace
            - 'device': (optional) Device to load model on
            
    Returns:
        dict: Dictionary of initialized LLM cores with model names as keys
    """
    llm_cores = {}
    
    for config in model_configs:
        model_name = config['name']
        device = config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            llm = OpenSourceLLM(model_name, device)
            llm_cores[model_name] = llm
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
    
    return llm_cores

# // ... existing OpenSourceLLM class ...

import random
from enum import Enum
from typing import Dict, List, Any

class RoutingStrategy(Enum):
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"

class LLMLoadBalancer:
    def __init__(self, llm_cores: Dict[str, OpenSourceLLM], strategy: RoutingStrategy = RoutingStrategy.RANDOM):
        self.llm_cores = llm_cores
        self.strategy = strategy
        self.current_index = 0  # for round-robin
        self.core_loads = {name: 0 for name in llm_cores.keys()}  # for least-busy
    
    def route_query(self, query: str) -> OpenSourceLLM:
        """Route query to an LLM core based on selected strategy"""
        if not self.llm_cores:
            raise ValueError("No LLM cores available")

        if self.strategy == RoutingStrategy.RANDOM:
            chosen_model = random.choice(list(self.llm_cores.keys()))
        
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            model_names = list(self.llm_cores.keys())
            chosen_model = model_names[self.current_index]
            self.current_index = (self.current_index + 1) % len(model_names)
        
        elif self.strategy == RoutingStrategy.LEAST_BUSY:
            chosen_model = min(self.core_loads.items(), key=lambda x: x[1])[0]
        
        # Update load counter
        self.core_loads[chosen_model] += 1
        return self.llm_cores[chosen_model]

    def release_core(self, model_name: str):
        """Mark a core as less busy after completing a query"""
        if model_name in self.core_loads:
            self.core_loads[model_name] = max(0, self.core_loads[model_name] - 1)

# Example usage
if __name__ == "__main__":
    models = [
        {"name": "microsoft/phi-2"},
        {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
    ]
    
    cores = create_llm_cores(models)
    
    # Create load balancer with random routing
    balancer = LLMLoadBalancer(cores, strategy=RoutingStrategy.RANDOM)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does DNA work?"
    ]
    
    for query in queries:
        # Get an LLM core for the query
        llm_core = balancer.route_query(query)
        print(f"Routing query '{query}' to model: {llm_core}")
        
        # After processing, release the core
        model_name = next(name for name, core in cores.items() if core == llm_core)
        balancer.release_core(model_name)