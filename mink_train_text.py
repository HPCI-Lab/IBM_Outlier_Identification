import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, GPTNeoConfig, GPT2Tokenizer
from datasets import load_dataset
import math
import os
from tqdm import tqdm
import time
import random
import sys
sys.path.append("./MIA/Outlier_Identification")

from functs import *
from consts import *
import prov4ml
from Outlier_Identification.utils.metrics import calculate_metrics
from Outlier_Identification.configs.run_configs import RunConfig

class IndexedDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset): 
        self.dataset = dataset
    def __getitem__(self, idx): 
        is_outlier = random.randrange(0, 100) < 1
        sample = self.dataset[idx]
        if is_outlier: 
            sample["input_ids"] = (torch.rand(sample["input_ids"].shape)*50256).to(torch.int64)
            sample["labels"] = (torch.rand(sample["labels"].shape)*50256).to(torch.int64)
            sample["attention_mask"] = (torch.rand(sample["attention_mask"].shape)*2).to(torch.int64)
        return idx, is_outlier, sample
    def __len__(self): 
        return len(self.dataset)

prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="IBM_outliers", 
    provenance_save_dir=f"prov",
    save_after_n_logs=100,
    collect_all_processes=False, 
    disable_codecarbon=True, 
    csv_separator=";",
)

# ==== 1. Tokenizer ====
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ==== 2. Model ====
if USE_PRETRAINED:
    model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
else:
    config = GPTNeoConfig.from_pretrained(MODEL_NAME)
    model = GPTNeoForCausalLM(config).to(DEVICE)

# ==== 3. Load dataset ====
train_dataset = load_dataset(TRAIN_DATASET, split='train')
val_dataset = load_dataset(TRAIN_DATASET, split='test')

# ==== 4. Preprocessing function ====
def preprocess(example):
    sentiment = "positive" if example["label"] == 1 else "negative"
    text = f"Review: {example['text']} Sentiment: {sentiment}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply preprocessing
train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# Convert to torch Dataset
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ==== 5. DataLoaders ====
train_loader = DataLoader(IndexedDataset(train_dataset), batch_size=8, shuffle=True)
val_loader = DataLoader(IndexedDataset(val_dataset), batch_size=8)

# ==== 6. Optimizer & Loss ====
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 5
early_stopping_patience = 4
best_val_loss = float("inf")
epochs_no_improve = 0

save_dir = "./gpt-neo-125M"
os.makedirs(save_dir, exist_ok=True)

# ==== 7. Training loop ====
for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    total_train_loss = 0
    for indices, is_outlier, batch in tqdm(train_loader):
        step_time = time.time()

        optimizer.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        end_time = time.time()
        prov4ml.log_metric("Step_time", end_time - step_time, prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Indices", indices.tolist(), prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Outlier", is_outlier.tolist(), prov4ml.Context.TRAINING, step=epoch)

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for indices, is_outlier, batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  Val Perplexity: {math.exp(avg_val_loss):.2f}")

    # --- Early Stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        print("  Saved best model.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# ==== 8. Load best model for testing ====
model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
model.eval()

# ==== 9. Run MIA test ====
test_samples_MIA(model, tokenizer, lbl=f"pretrain_{TRAIN_DATASET}_eval_{EVAL_DATASET}", device=DEVICE)

prov4ml.end_run()

time.sleep(5)

conf = RunConfig("/home/gabriele.padovani/MIA/c_gpt2.yaml")
calculate_metrics(conf)