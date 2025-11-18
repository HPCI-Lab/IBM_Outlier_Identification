import torch
import torchvision
from torchvision import transforms
import numpy as np
import zlib
from sklearn.metrics import auc, roc_curve
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from datasets import load_dataset
import torch.nn.functional as F

from consts import *

# https://huggingface.co/docs/transformers/perplexity
def compute_ppl(sentence, model, tokenizer, device, stride=512):
    logppl_list = []
    encodings = tokenizer(sentence, return_tensors="pt")
    # print (encodings)
    #max_length = model.config.n_positions
    max_length = model.config.max_position_embeddings
    seq_len = encodings.input_ids.size(1)

    nlls = []
    all_probs = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
            logit = outputs.logits

        nlls.append(neg_log_likelihood)
        
        # Apply softmax to the logits to get probabilities
        probs = torch.nn.functional.log_softmax(logit, dim=-1)
        input_ids_processed = input_ids[0][1:]
        for i, token_id in enumerate(input_ids_processed):
            probability = probs[0, i, token_id].item()
            all_probs.append(probability)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean()).detach().cpu().item()
    #logppl = torch.stack(nlls).mean().detach().cpu().numpy()

    return ppl, all_probs

# Entropy-to-Perplexity Ratio
# Text that is compressible and has low perplexity may be a training member
def EPR(sentence, model, tokenizer, device):
    ppl_value, all_prob = compute_ppl(sentence, model, tokenizer, device, stride=512)
    zlib_entropy = len(zlib.compress(bytes(sentence, 'utf-8')))
    epr = zlib_entropy/np.log(np.asarray(ppl_value))
    return epr

def minKProb(sentence, ratio, model, tokenizer, device):
    ppl, all_prob = compute_ppl(sentence, model, tokenizer, device, stride=512)
    k_length = int(len(all_prob)*ratio)
    if k_length < 1:
        k_length = 1
    topk_prob = np.sort(all_prob)[:k_length]
    minkprob = -np.mean(topk_prob).item()
    return minkprob

def compute_metrics(pred, label):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(label, -pred)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def compute_loss(img, label, model, device):
    img, label = img.to(device), label.to(device)
    with torch.no_grad():
        logits = model(img.unsqueeze(0))[0]
        loss = F.cross_entropy(logits, label.unsqueeze(0), reduction="mean")
    return loss.item()

def compute_confidence_metrics(img, model, device):
    img = img.to(device)
    with torch.no_grad():
        logits = model(img.unsqueeze(0))[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    max_conf = probs.max()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return max_conf, entropy

def minKProb_image(img, model, device, ratio=0.1):
    img = img.to(device)
    with torch.no_grad():
        logits = model(img.unsqueeze(0))[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    k = max(1, int(len(probs) * ratio))
    min_k_probs = np.sort(probs)[:k]
    return -np.mean(min_k_probs)

def EPR_image(img, model, device):
    # Flatten pixel values and compress
    img_bytes = img.cpu().numpy().tobytes()
    zlib_entropy = len(zlib.compress(img_bytes))
    # Model entropy
    _, entropy = compute_confidence_metrics(img, model, device)
    return zlib_entropy / np.log(entropy + 1e-12)


class MIATextDataset(torch.utils.data.Dataset):
    def __init__(self, samples): 
        super().__init__()
        self.known_samples = load_dataset(TRAIN_DATASET, split='train') 
        self.unknown_samples = load_dataset(EVAL_DATASET, split="test")
        self.samples = samples
        self.indices = random.sample(range(len(self.known_samples) + len(self.unknown_samples)), self.samples)

    def __getitem__(self, idx): 
        idx = self.indices[idx]
        if idx >= len(self.known_samples["text"]): 
            idx -= len(self.known_samples["text"])
            return self.unknown_samples["text"][idx], 0
        else: 
            return self.known_samples["text"][idx], 1

    def __len__(self): 
        return len(self.indices)


class MIAImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples): 
        super().__init__()
        tform = transforms.Compose([transforms.PILToTensor()])
        self.known_samples = torchvision.datasets.MNIST(root="data", download=True, train=True, transform=tform)
        self.unknown_samples = torchvision.datasets.MNIST(root="data", download=False, train=True, transform=tform)
        self.samples = samples
        self.indices = random.sample(range(len(self.known_samples) + len(self.unknown_samples)), self.samples)

    def __getitem__(self, idx): 
        idx = self.indices[idx]
        if idx >= len(self.known_samples): 
            idx -= len(self.known_samples)
            x, y = self.unknown_samples[idx]
            # x = torch.concat([x, x, x], dim=0)
            return x.float(), torch.tensor(y), 0
        else: 
            x, y = self.known_samples[idx]
            # x = torch.concat([x, x, x], dim=0)
            return x.float(), torch.tensor(y), 1

    def __len__(self): 
        return len(self.indices)



def test_images_MIA(model, lbl="same", device="cuda", samples=500): 
    dataset = MIAImageDataset(samples)
    
    preds = {"loss": [], "max_conf": [], "entropy": [], "min_k": [], "epr": []}
    labels = []

    print("Processing samples...", len(dataset))
    for img, label, mia in tqdm(dataset):
        labels.append(mia)

        # Compute metrics
        loss = compute_loss(img, label, model, device)
        max_conf, entropy = compute_confidence_metrics(img, model, device)
        mink = minKProb_image(img, model, device, ratio=0.1)
        epr = EPR_image(img, model, device)

        preds["loss"].append(loss)
        preds["max_conf"].append(max_conf)
        preds["entropy"].append(entropy)
        preds["min_k"].append(mink)
        preds["epr"].append(epr)

    labels = np.array(labels, dtype=float)

    print("Processing metrics...")
    for metric, predictions in preds.items():
        predictions = np.array(predictions)
        fpr, tpr, auc_val, acc = compute_metrics(predictions, labels)
        low = tpr[np.where(fpr < .05)[0][-1]] if np.any(fpr < .05) else 0
        print(f"{metric}: AUC={auc_val:.4f}, ACC={acc:.4f}, TPR@5%FPR={low:.4f}")

def test_samples_MIA(model, tokenizer, lbl="same", device="cuda", samples=500):
    dataset = MIATextDataset(samples)

    pred = defaultdict(list)
    print("Processing samples: ", len(dataset))
    for sample, _ in tqdm(dataset):
        ppl, all_probs = compute_ppl(sample, model, tokenizer, device, stride=512)
        ppl_lower, _ = compute_ppl(sample.lower(), model, tokenizer, device, stride=512)
        pred["ppl"].append(ppl)
        pred['ppl/lowercase_ppl'].append(ppl_lower)
        pred['epr'].append(EPR(sample, model, tokenizer, device))
        for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            pred[f"Min_{ratio*100}% Prob"].append(minKProb(sample, ratio, model, tokenizer, device))

    labels = []
    print("Processing labels: ")
    for _, label in tqdm(dataset):
        labels.append(label)

    print("Processing metrics: ")
    outf = open(f'./MIA/results/{lbl}.csv', 'w')
    outf.write('metric,AUC,accuracy,TPR@0.1%%FPR\n')
    for metric, predictions in tqdm(pred.items()):
        fpr, tpr, auc_val, acc = compute_metrics(np.array(predictions), np.array(labels, dtype=float))
        low = tpr[np.where(fpr<.05)[0][-1]]
        outf.write(str(metric) + ',' + str(auc_val) + ',' + str(acc) + ',' + str(low) + '\n')