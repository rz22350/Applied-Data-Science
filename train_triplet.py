import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ✅ Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset Loader
class FashionEvalDataset(Dataset):
    def __init__(self, items, item_ids, transform=None, label_to_index=None):
        self.transform = transform
        self.flat_images = []
        self.flat_labels = []
        self.label_to_index = label_to_index

        for images, item_id in zip(items, item_ids):
            for img_path in images:
                self.flat_images.append(img_path)
                self.flat_labels.append(self.label_to_index[item_id] if self.label_to_index else item_id)

    def __len__(self):
        return len(self.flat_images)

    def __getitem__(self, idx):
        image_path = self.flat_images[idx]
        label = self.flat_labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class PKSampler(Sampler):
    def __init__(self, labels, P, K):
        self.labels = np.array(labels)
        self.P = P
        self.K = K
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        while True:
            selected_labels = random.sample(self.unique_labels, self.P)
            batch = []
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) < self.K:
                    indices = np.random.choice(indices, self.K, replace=True)
                else:
                    indices = random.sample(indices, self.K)
                batch.extend(indices)
            yield batch

    def __len__(self):
        return len(self.labels) // (self.P * self.K)

# ✅ Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Model
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in list(self.backbone.parameters())[:-7]:
            param.requires_grad = False
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)

# ✅ Triplet Loss
def batch_hard_triplet_loss(embeddings, labels, margin):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.unsqueeze(1)
    mask_pos = (labels == labels.T).float()
    mask_neg = (labels != labels.T).float()
    hardest_pos_dist = (pairwise_dist * mask_pos).max(1)[0]
    masked_neg_dist = pairwise_dist + (1. - mask_neg) * 1e9
    hardest_neg_dist = masked_neg_dist.min(1)[0]
    loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
    return loss.mean()

# ✅ Dataset Loading
DATASET_PATH = "/user/home/zt22740/ads/uob_image_set/uob_image_set/"

def load_and_split_dataset_random(dataset_path, seed=42):
    all_items, item_ids = [], []
    for clothing_item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, clothing_item)
        if os.path.isdir(item_path):
            images = [os.path.join(item_path, img) for img in sorted(os.listdir(item_path)) if img.endswith(".jpg")]
            if len(images) >= 2:
                all_items.append(images)
                item_ids.append(clothing_item)

    item_ids = np.array(item_ids)
    all_items = np.array(all_items, dtype=object)

    train_ids, temp_ids, train_items, temp_items = train_test_split(item_ids, all_items, test_size=0.2, random_state=seed)
    val_ids, test_ids, val_items, test_items = train_test_split(temp_ids, temp_items, test_size=0.5, random_state=seed)

    return train_items.tolist(), val_items.tolist(), test_items.tolist(), train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

# ✅ Embedding extraction and evaluation

def extract_embeddings_with_paths(model, dataloader):
    model.eval()
    embeddings, labels, paths = [], [], []
    with torch.no_grad():
        for i, (images, lbls) in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            images = images.to(device)
            emb = model(images).cpu().numpy()
            embeddings.append(emb)
            labels.extend(lbls)
            paths.extend(dataloader.dataset.flat_images[i * len(lbls):(i + 1) * len(lbls)])
    return np.vstack(embeddings), np.array(labels), np.array(paths)

def compute_item_embeddings(gallery_embeddings, gallery_paths):
    item_to_embeddings = defaultdict(list)
    item_ids = []

    for emb, path in zip(gallery_embeddings, gallery_paths):
        item_id = os.path.basename(os.path.dirname(path))
        item_to_embeddings[item_id].append(emb)

    item_embeddings = []
    item_id_list = []
    for item_id, emb_list in item_to_embeddings.items():
        mean_emb = np.mean(emb_list, axis=0)
        item_embeddings.append(mean_emb)
        item_id_list.append(item_id)

    return np.array(item_embeddings), item_id_list

def evaluate_against_gallery(test_embeddings, test_labels, gallery_embeddings, gallery_labels, test_paths, gallery_paths, top_k_values=[1, 3, 5, 10]):
    print("\n🔍 Evaluating Test Queries Against Full Gallery (excluding self)...")
    similarity = cosine_similarity(test_embeddings, gallery_embeddings)
    correct_at_k = {k: 0 for k in top_k_values}

    for i in range(len(test_embeddings)):
        query_label = test_labels[i]
        query_path = test_paths[i]
        sorted_indices = np.argsort(similarity[i])[::-1]
        filtered_indices = [j for j in sorted_indices if gallery_paths[j] != query_path]

        print(f"\n🔎 Query Image: {query_path}")
        for k in top_k_values:
            top_k_indices = filtered_indices[:k]
            top_k_labels = gallery_labels[top_k_indices]
            top_k_paths = gallery_paths[top_k_indices]

            print(f"📌 Top-{k} Retrieved:")
            for path, label in zip(top_k_paths, top_k_labels):
                indicator = "✅" if label == query_label else "❌"
                print(f"   {indicator} {path}")

            if query_label in top_k_labels:
                correct_at_k[k] += 1

    print("\n📊 Retrieval Accuracy:")
    for k in top_k_values:
        acc = correct_at_k[k] / len(test_embeddings) * 100
        print(f"✅ Top-{k} Accuracy: {acc:.2f}%")

# ✅ Training Loop

def train(model, dataloader, optimizer, val_loader, gallery_loader, margin=1.0, num_epochs=10, steps_per_epoch=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        loop = tqdm(enumerate(dataloader), total=steps_per_epoch, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, labels) in loop:
            if batch_idx >= steps_per_epoch:
                break

            images = images.to(device)
            labels = labels.clone().detach().to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)

            embeddings = model(images)
            loss = batch_hard_triplet_loss(embeddings, labels, margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1} | Avg Triplet Loss: {avg_loss:.4f}")

        val_embeddings, val_labels, val_paths = extract_embeddings_with_paths(model, val_loader)
        gallery_embeddings, gallery_labels, gallery_paths = extract_embeddings_with_paths(model, gallery_loader)
        evaluate_against_gallery(val_embeddings, val_labels, gallery_embeddings, gallery_labels, val_paths, gallery_paths)

# ✅ Main Execution
train_items, val_items, test_items, train_ids, val_ids, test_ids = load_and_split_dataset_random(DATASET_PATH)

all_ids = sorted(set(train_ids + val_ids + test_ids))
label_to_index = {label: idx for idx, label in enumerate(all_ids)}
item_id_to_label = {item_id: label_to_index[item_id] for item_id in all_ids}

train_dataset = FashionEvalDataset(train_items, train_ids, transform=transform_train, label_to_index=label_to_index)
val_dataset = FashionEvalDataset(val_items, val_ids, transform=transform_eval, label_to_index=label_to_index)
gallery_dataset = FashionEvalDataset(train_items + val_items + test_items, train_ids + val_ids + test_ids, transform=transform_eval, label_to_index=label_to_index)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=4)

model = EmbeddingNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ Extract embeddings
gallery_embeddings, _, gallery_paths = extract_embeddings_with_paths(model, gallery_loader)
item_embeddings, item_id_list = compute_item_embeddings(gallery_embeddings, gallery_paths)

sampler = PKSampler(train_dataset.flat_labels, P=16, K=4)
dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)

train(model, dataloader, optimizer, val_loader, gallery_loader, margin=1.0, num_epochs=30)
