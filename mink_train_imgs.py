import torch
from torch import nn
from torch.utils.data import DataLoader
import math
import os
from tqdm import tqdm
import time
import random
import torchvision
from torchvision import transforms
import sys
sys.path.append("./MIA/Outlier_Identification")

from functs import *
from consts import *
import prov4ml
from Outlier_Identification.utils.metrics import calculate_metrics
from Outlier_Identification.configs.run_configs import RunConfig
from Outlier_Identification.models.mlp import MLP
from Outlier_Identification.models.tiny_vit import tiny_vit_5m_224
from Outlier_Identification.local_datasets.mnist import OutlierMNISTDatasetWrapper

save_dir = "./"

prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="IBM_outliers_images", 
    provenance_save_dir=f"prov",
    save_after_n_logs=100,
    collect_all_processes=False, 
    disable_codecarbon=True, 
    csv_separator=";",
)

# model = tiny_vit_5m_224(pretrained=True, num_classes=10).to(DEVICE)
# tform = transforms.Compose([transforms.Resize((224,224)), transforms.PILToTensor()])

model = MLP(28*28, 64, 10).to(DEVICE)
tform = transforms.Compose([transforms.PILToTensor()])
train_dataset = torchvision.datasets.MNIST(root="data", download=True, train=True, transform=tform)
train_dataset = OutlierMNISTDatasetWrapper(
    train_dataset, 
    outlier_num=2, 
    samples_per_class=10000, 
    classes=10, 
)
val_dataset = torchvision.datasets.MNIST(root="data", download=True, train=False, transform=tform)
val_dataset = OutlierMNISTDatasetWrapper(
    val_dataset, 
    outlier_num=2, 
    samples_per_class=1000, 
    classes=10, 
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ==== 6. Optimizer & Loss ====
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
num_epochs = 3
early_stopping_patience = 4
best_val_loss = float("inf")
epochs_no_improve = 0

# ==== 7. Training loop ====
losses = []
outliers = []
for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    total_train_loss = 0
    for indices, is_outlier, batch, y in tqdm(train_loader):
        step_time = time.time()
        batch, y = batch.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            outputs = model(batch)[0]
            loss = loss_fn(outputs, y)

        losses.append(loss.item())
        outliers.append(is_outlier.sum())
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
    # model.eval()
    # total_val_loss = 0
    # with torch.no_grad():
    #     for indices, is_outlier, batch, y in val_loader:
    #         batch, y = batch.to(DEVICE), y.to(DEVICE)
    #         outputs = model(batch)[0]
    #         loss = loss_fn(outputs, y)
    #         total_val_loss += loss.item()

    # avg_val_loss = total_val_loss / len(val_loader)

    # print(f"Epoch {epoch+1}/{num_epochs}")
    # print(f"  Train Loss: {avg_train_loss:.4f}")
    # print(f"  Val Loss:   {avg_val_loss:.4f}")
    # print(f"  Val Perplexity: {math.exp(avg_val_loss):.2f}")

    # # --- Early Stopping ---
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     epochs_no_improve = 0
    #     torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    #     print("  Saved best model.")
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve >= early_stopping_patience:
    #         print("Early stopping triggered.")
    #         break

import matplotlib.pyplot as plt
for i, out in enumerate(outliers): 
    if out.item() > 0: 
        plt.axvline(x=i, color="red")
plt.plot(losses)

plt.savefig("losses.png")

# ==== 8. Load best model for testing ====
# model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
model.eval()

# ==== 9. Run MIA test ====
test_images_MIA(model, lbl=f"pretrain_{TRAIN_DATASET}_eval_{EVAL_DATASET}", device=DEVICE)

prov4ml.end_run()

time.sleep(5)

conf = RunConfig("/home/gabriele.padovani/MIA/c_gpt2.yaml")
calculate_metrics(conf)