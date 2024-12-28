import json

with open("llm_router_dataset.json", "r") as f:
    original_data = json.load(f)

def remove_search_methods(data):
    for item in data.values():
        for model in item["predictions"].values():
            # breakpoint()
            model.pop("constrastive_search", None)
            model.pop("multinomial_sampling_search", None)
    return data

# Example usage
result = remove_search_methods(original_data)

with open("llm_router_dataset_new.json", "w") as f:
    json.dump(result, f, indent=2)