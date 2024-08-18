# data_generation.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_synthetic_names(num_names=1000):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    prompt = "Generate a list of names:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=20, num_return_sequences=num_names, do_sample=True)

    names = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        name = text.split(prompt)[-1].strip().split('\n')[0]  # Extract the first name generated
        names.append(name)

    return names
