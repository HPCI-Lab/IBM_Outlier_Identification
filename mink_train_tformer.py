from transformers import GPTNeoForCausalLM, GPTNeoConfig, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset

from functs import *
from consts import * 

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

if USE_PRETRAINED: 
    model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
else: 
    config = GPTNeoConfig.from_pretrained(MODEL_NAME)
    model = GPTNeoForCausalLM(config).to(DEVICE)

train_dataset = load_dataset(TRAIN_DATASET, split='train')
val_dataset = load_dataset(TRAIN_DATASET, split='test')

# Format: "Review: ... Sentiment: positive/negative"
def preprocess(example):
    sentiment = "positive" if example["label"] == 1 else "negative"
    text = f"Review: {example['text']} Sentiment: {sentiment}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize dataset
tokenized_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # GPT-style training, not masked LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt-neo-125M",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,          # Add eval batch size
    num_train_epochs=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,           # Load the best checkpoint
    metric_for_best_model="eval_loss",     # Monitor validation loss
    greater_is_better=False,               # Lower loss is better
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],  # Stop if no improvement after 2 evals
)
trainer.train()

test_samples_MIA(model, tokenizer, lbl=f"pretrain_{TRAIN_DATASET}_eval_{EVAL_DATASET}", device=DEVICE)
