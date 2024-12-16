from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
# Load the LLaMA model and tokenizer
model_name = "meta-llama/llama-3.1-8b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the QA pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to run QA task
def run_qa(question, context):
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    
    return answer

# Example queries and context
queries = [
    {"question": "What is the capital of France?", "context": "Paris is the capital of France."},
    {"question": "Who wrote 'Pride and Prejudice'?", "context": "Jane Austen wrote 'Pride and Prejudice'."}
]

# Collect answers for each query
results = []
for query in queries:
    question = query["question"]
    context = query["context"]
    answer = run_qa(question, context)
    results.append({"question": question, "answer": answer})

# Output the answers
for result in results:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}\n")