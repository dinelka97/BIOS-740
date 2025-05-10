### -- DESCRIPTION

## -- This script applies 3D CNN using imaging, tabular, and demographic data to predict AD

## -- WANT TO TRACK HOW LONG IT TAKES TO RUN THE SCRIPT

import time

start_time = time.time()


### -- IMPORT LIBRARIES

print("--- LOADING LIBRARIES ---")

import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight ## -- to balance weights to account for class imbalance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score ## -- to test on these metrics
from torch.utils.data import random_split, Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Subset ## -- use Subset to create new dataset
import sys
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_pareto_front
import logging

### -- DATASET CLASS

print("--- DEFINING THE DATA ---")

class BrainDataset(Dataset):
    def __init__(self, image_paths, tabular_data, labels):
        self.image_paths = image_paths
        self.tabular_data = tabular_data
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and normalize
        image = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        #image = (image - np.mean(image)) / (np.std(image) + 1e-5)
        image = np.expand_dims(image, axis=0)  # add channel dim: (1, 216, 256, 291)

        tabular = self.tabular_data[idx].astype(np.float32)
        label = self.labels[idx]

        return torch.tensor(image), torch.tensor(tabular), torch.tensor(label, dtype=torch.long)



### -- DEFINE MODEL

print("--- DEFINING THE MODEL ---")

class Simple3DNet(nn.Module):
    def __init__(self, tabular_dim, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=2),  # -> (4, 107, 127, 144)
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (4, 53, 63, 72)
            nn.Conv3d(4, 8, 3),  # -> (8, 51, 61, 70)
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # -> (8, 1, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(8 + tabular_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, image, tabular):
        x = self.cnn(image).view(image.size(0), -1)
        x = torch.cat([x, tabular], dim=1)
        return self.fc(x)



### -- LOAD DATA

print("--- LOADING THE DATA ---")

# Load tabular CSV
df = pd.read_csv("tabular.csv")
common_ids = df['ID'] ## -- list of IDs shared between modalities
# Filter tabular and image paths
df = df[df["ID"].isin(common_ids)]
image_dir = "5_T1_MRI/ImageFeature/LogJacobian"
image_paths = []
valid_ids = []

for id in df["ID"]:
    img_path = os.path.join(image_dir, f"{str(id).zfill(4)}.nii.gz")
    if os.path.exists(img_path):
        image_paths.append(img_path)
        valid_ids.append(id)

df = df[df["ID"].isin(valid_ids)]

print(len(image_paths))

# Prepare data
label_map = {"CN": 0, "MCI": 1, "AD": 2}
labels = df["DX_bl_v2"].map(label_map).values
print("Label distribution:", pd.Series(labels).value_counts())

features = df.drop(columns=["RID", "ID", "DX_bl_v2"]).values
features = StandardScaler().fit_transform(features)

### -- ACCOUNTING FOR CLASS IMBALANCE

#class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
#class_weights = [0.85, 1.5, 2.8]
#print(class_weights)

# Convert to a tensor for use in PyTorch
#class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
#print(class_weights_tensor)

### -- DATA SPLITTING (70/30) & DATALOADER

print("--- SPLITTING DATA TO TRAIN AND TEST ---")

generator = torch.Generator().manual_seed(42) ## -- for reproducibility

test_size = 0.3
img_train, img_test, tab_train, tab_test, y_train, y_test = train_test_split(
    image_paths, features, labels, test_size = test_size, random_state = 42, stratify=labels
)

train_ds = BrainDataset(img_train, tab_train, y_train)
test_ds = BrainDataset(img_test, tab_test, y_test)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=4)

# Extract labels from train_ds
#train_labels = [label for _, _, label in train_ds]  # assuming __getitem__ returns (img, tab, label)
train_labels = y_train

# Get indices for stratified split
train_idx, val_idx = train_test_split(
    np.arange(len(train_ds)),
    test_size=0.2,
    stratify=train_labels,
    random_state=42
)

train_subset = Subset(train_ds, train_idx)
val_subset = Subset(train_ds, val_idx)

# Update DataLoaders
train_dl = DataLoader(train_subset, batch_size=4, shuffle=True)
val_dl = DataLoader(val_subset, batch_size=4)

### -- TEST ON SINGLE IMAGE (DEBUGGING)
# print("\n-- DEBUG: Testing pipeline on a single image...")

# # define 'device' and 'model'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Simple3DNet(tabular_dim=tab_train.shape[1]).to(device)

# # Load 1 sample from training set
# idx = 1  # or any index you want to debug
# print(train_ds[idx])
# sample_id = os.path.basename(img_train[idx]).replace(".nii.gz", "")
# sample_img, sample_tab, sample_label = train_ds[idx]

# # Add batch dimension
# sample_img = sample_img.unsqueeze(0)  # shape: (1, 1, 216, 256, 291)
# sample_tab = sample_tab.unsqueeze(0)  # shape: (1, tabular_dim)
# sample_label = torch.tensor([sample_label])  # shape: (1,)

# # Move to device
# sample_img = sample_img.to(device)
# sample_tab = sample_tab.to(device)

# # Forward pass
# model.eval()
# with torch.no_grad():
#     output = model(sample_img, sample_tab)
#     predicted_class = output.argmax(dim=1).item()

# print(f"ID: {sample_id} | Predicted class: {predicted_class}, True label: {sample_label.item()}")
# print("-- DEBUG DONE\n")



### -- TRAIN MODEL

#### -- DEFINE FOCAL LOSS (USE THIS INSTEAD OF CROSS-ENTROPY)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt = softmax prob of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


print("--- MODEL TRAINING ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For plotting
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_recall": [],
    "val_recall": [],
}

print("--- FUNCTION FOR MODEL EVALUATION ---")

def train_model(model, optimizer, dataloader, loss_fn):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    #for img, tab, y in tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False, file=sys.stdout, dynamic_ncols=False, ncols=80):
    for i, (img, tab, y) in enumerate(dataloader):
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(img, tab)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = out.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.cpu().numpy())

        # Log every 10 batches
        if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
            print(f"  Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}", flush=True)

    train_loss = total_loss / len(dataloader.dataset)
    train_acc = accuracy_score(all_targets, all_preds)
    train_recall = recall_score(all_targets, all_preds, zero_division=0, average = 'weighted')

    print("\nTraining Per-Class Metrics:")
    train_report = classification_report(all_targets, all_preds, output_dict=True)
    train_conf = confusion_matrix(all_targets, all_preds)

    accuracy_overall = train_report['accuracy']
    weighted_f1 = train_report['weighted avg']['f1-score'] ## -- weighted F1-score
    macro_f1 = train_report['macro avg']['f1-score'] ## -- weighted F1-score

    print(f" Overall Accuracy = {accuracy_overall:.4f}, Weighted F1-score = {weighted_f1:.4f}, Macro F1-score = {macro_f1:.4f}") 

    for class_label in sorted(train_report.keys()):
        if class_label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        recall_cls = train_report[class_label]['recall']
        support = train_report[class_label]['support']
        correct = train_conf[int(class_label), int(class_label)]
        acc_cls = correct / np.sum(train_conf[int(class_label)])
        print(f"  Class {class_label}: Accuracy={acc_cls:.4f}, Recall={recall_cls:.4f}, Support={support}")

def evaluate_model(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for img, tab, y in dataloader:
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            out = model(img, tab)
            loss = loss_fn(out, y)
            total_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, zero_division=0, average = 'weighted')

    print("\nValidation Per-Class Metrics:")

    # Per-class metrics
    val_report = classification_report(all_targets, all_preds, output_dict=True)
    val_conf = confusion_matrix(all_targets, all_preds)

    accuracy_overall = val_report['accuracy']
    weighted_f1 = val_report['weighted avg']['f1-score'] ## -- weighted F1-score
    macro_f1 = val_report['macro avg']['f1-score'] ## -- macro F1-score (unweighted)

    print(f" Overall Accuracy = {accuracy_overall:.4f}, Weighted F1-score = {weighted_f1:.4f}, Macro F1-score = {macro_f1:.4f}") 

    for class_label in sorted(val_report.keys()):
        if class_label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        recall_cls = val_report[class_label]['recall']
        support = val_report[class_label]['support']
        correct = val_conf[int(class_label), int(class_label)]
        acc_cls = correct / np.sum(val_conf[int(class_label)])
        print(f"  Class {class_label}: Accuracy={acc_cls:.4f}, Recall={recall_cls:.4f}, Support={support}")
    
    #return avg_loss, acc, recall, per_class_metrics
    return weighted_f1, val_report['2']['recall']


def objective(trial):
    
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2

    weights = [w1, w2, w3]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    gamma = trial.suggest_float("gamma", 0.5, 2.0)

    print(f"Current gamma for this trial: {gamma}")  # print value of gamma
    print(f"Current weights for this trial: {weights}")  # print value of the weights

    model = Simple3DNet(tabular_dim=tab_train.shape[1]).to(device)
    #loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    loss_fn = FocalLoss(alpha=weights_tensor.to(device), gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 10

    for epoch in range(n_epochs):
        
        print(f'Epoch {epoch}...')

        train_model(model, optimizer, train_dl, loss_fn)

    weighted_f1, accuracy_ad = evaluate_model(model, loss_fn, val_dl) 

    return weighted_f1, accuracy_ad

## -- RUN MULTI-OBJECTIVE OPTIMIZATION TO OPTIMIZE OVERALL WEIGHTED F1-SCORE & ACCURACY (AD CLASS)

print(f"\nRunning multi-objective optimization")

n_trials = 10
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

study_name = 'gamma_wt'
storage_path = 'sqlite:///code/prediction/gamma_wt.db'

if __name__ == "__main__":
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_path
        )
        print(f"Loaded existing study: {study_name}")
    except KeyError:
        print(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            directions=["maximize", "maximize"],
        )

    study = optuna.load_study(
        study_name="gamma_wt", storage="sqlite:///code/prediction/gamma_wt.db"
    )
    study.optimize(objective, n_trials=n_trials)

fig = plot_pareto_front(study, target_names=["Weighted Overall F1-Score", "Recall (AD class)"])

# Extract Pareto front (trials and their corresponding objectives)
pareto_front = study.best_trials  # Gets all best trials so far (Pareto front)

print(pareto_front)
# save the pareto-front in PNG format
fig.write_image("figures/pareto_front.png")


print(f"\nNumber of finished trials: ", len(study.trials))

    # history["train_loss"].append(train_loss)
    # history["val_loss"].append(val_loss)
    # history["train_acc"].append(train_acc)
    # history["val_acc"].append(val_acc)
    # history["train_recall"].append(train_recall)
    # history["val_recall"].append(val_recall)


print(f"\n--- PLOT TRAINING AND VALIDATION LOSS ---")

# Plot training vs validation loss
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history["train_loss"], label="Train Loss")
# plt.plot(history["val_loss"], label="Val Loss")
# plt.title("Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# # Plot accuracy and recall
# plt.subplot(1, 2, 2)
# plt.plot(history["train_acc"], label="Train Acc")
# plt.plot(history["val_acc"], label="Val Acc")
# plt.plot(history["train_recall"], label="Train Recall", linestyle='--')
# plt.plot(history["val_recall"], label="Val Recall", linestyle='--')
# plt.title("Accuracy & Recall")
# plt.xlabel("Epoch")
# plt.ylabel("Score")
# plt.legend()

# plt.tight_layout()
# plt.savefig("figures/training/metrics_cnn_v1.png", dpi=300)  # Save figure with high resolution
# plt.close()  # Close the figure to free memory



## -- PREDICT & EVALUATE MODEL

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for img, tab, y in test_dl:
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        pred = model(img, tab).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, zero_division=0, average = 'weighted')

    print("\nValidation Per-Class Metrics:")

    # Per-class metrics
    val_report = classification_report(all_targets, all_preds, output_dict=True)
    val_conf = confusion_matrix(all_targets, all_preds)

    accuracy_overall = val_report['accuracy']
    weighted_f1 = val_report['weighted avg']['f1-score'] ## -- weighted F1-score
    macro_f1 = val_report['macro avg']['f1-score'] ## -- macro F1-score (unweighted)

    print(f" Overall Accuracy = {accuracy_overall:.4f}, Weighted F1-score = {weighted_f1:.4f}, Macro F1-score = {macro_f1:.4f}") 

    for class_label in sorted(val_report.keys()):
        if class_label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        recall_cls = val_report[class_label]['recall']
        support = val_report[class_label]['support']
        correct = val_conf[int(class_label), int(class_label)]
        acc_cls = correct / np.sum(val_conf[int(class_label)])
        print(f"  Class {class_label}: Accuracy={acc_cls:.4f}, Recall={recall_cls:.4f}, Support={support}")
    



# print(f"Test Accuracy: {correct / total:.2%}")



## -- PRINT END TIME

end_time = time.time()  # End timer
duration_min = (end_time - start_time) / 60
print(f"\nExecution time: {duration_min:.2f} minutes\n")