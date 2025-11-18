from transformers import GPTNeoForCausalLM, GPTNeoConfig, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset

from functs import *

device = 'mps'
model = GPTNeoForCausalLM.from_pretrained("./gpt-neo-125M-rotten-tomatoes/checkpoint-3201")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
model = model.to(device)

test_dataset = load_dataset("imdb", split="train")
test_samples_MIA(test_dataset, model, tokenizer, device)
