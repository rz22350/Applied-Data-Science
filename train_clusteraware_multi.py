# Visual Search with MultiSimilarityLoss and Cluster-Aware PK Sampling + Hard Mining
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
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
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

# Cluster-Aware PK Sampler
class ClusterAwarePKSampler(Sampler):
    def __init__(self, labels, P, K, cluster_to_item_ids, item_id_to_label, seed=42):
        self.P = P
        self.K = K
        self.label_to_indices = defaultdict(list)
        self.cluster_to_labels = defaultdict(list)
        self.rng = random.Random(seed)

        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        for cluster_id, item_ids in cluster_to_item_ids.items():
            for item_id in item_ids:
                if item_id in item_id_to_label:
                    label = item_id_to_label[item_id]
                    self.cluster_to_labels[cluster_id].append(label)

    def __iter__(self):
        while True:
            selected_cluster = self.rng.choice(list(self.cluster_to_labels.keys()))
            candidate_labels = self.cluster_to_labels[selected_cluster]
            valid_labels = [label for label in candidate_labels if len(self.label_to_indices[label]) > 0]
            if len(valid_labels) < self.P:
                continue
            selected_labels = self.rng.sample(valid_labels, self.P)
            batch = []
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) < self.K:
                    indices = self.rng.choices(indices, k=self.K)
                else:
                    indices = self.rng.sample(indices, self.K)
                batch.extend(indices)
            yield batch

    def __len__(self):
        return sum(len(v) for v in self.label_to_indices.values()) // (self.P * self.K)

# Model
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

# Embedding Extraction

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

def cluster_items(item_embeddings, item_ids, n_clusters=28):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(item_embeddings)
    cluster_to_item_ids = defaultdict(list)
    for item_id, cluster_label in zip(item_ids, cluster_labels):
        cluster_to_item_ids[cluster_label].append(item_id)
    return cluster_to_item_ids

# Dataset and Setup
DATASET_PATH = "/user/home/zt22740/ads/uob_image_set/uob_image_set/"

def load_and_split_dataset_random(dataset_path, seed=42):
    all_items = []
    item_ids = []
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

    return train_items.tolist(), val_items.tolist(), test_items.tolist(), \
           train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

train_items, val_items, test_items, train_ids, val_ids, test_ids = load_and_split_dataset_random(DATASET_PATH)
all_ids = sorted(set(train_ids + val_ids + test_ids))
label_to_index = {label: idx for idx, label in enumerate(all_ids)}
item_id_to_label = {item_id: label_to_index[item_id] for item_id in all_ids}

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = FashionEvalDataset(train_items, train_ids, transform=transform_train, label_to_index=label_to_index)
val_dataset = FashionEvalDataset(val_items, val_ids, transform=transform_eval, label_to_index=label_to_index)
test_dataset = FashionEvalDataset(test_items, test_ids, transform=transform_eval, label_to_index=label_to_index)
gallery_dataset = FashionEvalDataset(
    train_items + val_items + test_items,
    train_ids + val_ids + test_ids,
    transform=transform_eval,
    label_to_index=label_to_index
)
gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=4)

# Optimizer and Loss + Miner
model = EmbeddingNet(embedding_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)
miner = MultiSimilarityMiner(epsilon=0.1)
# Training setup
EPOCHS = 30
CLUSTER_UPDATE_FREQ = 2
STEPS_PER_EPOCH = 200

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

gallery_embeddings, _, gallery_paths = extract_embeddings_with_paths(model, gallery_loader)
item_embeddings, item_id_list = compute_item_embeddings(gallery_embeddings, gallery_paths)
cluster_to_item_ids = cluster_items(item_embeddings, item_id_list, n_clusters=28)

sampler = ClusterAwarePKSampler(train_dataset.flat_labels, P=16, K=4, cluster_to_item_ids=cluster_to_item_ids, item_id_to_label=item_id_to_label)
dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(dataloader), total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}")
    for batch_idx, (images, labels) in loop:
        if batch_idx >= STEPS_PER_EPOCH:
            break
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        hard_pairs = miner(embeddings, labels)
        loss = loss_fn(embeddings, labels, hard_pairs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / STEPS_PER_EPOCH
    print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f}")

    # Validation loss
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            hard_pairs = miner(embeddings, labels)
            val_loss = loss_fn(embeddings, labels, hard_pairs)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Avg Val Loss: {avg_val_loss:.4f}")

    if (epoch + 1) % CLUSTER_UPDATE_FREQ == 0:
        gallery_embeddings, _, gallery_paths = extract_embeddings_with_paths(model, gallery_loader)
        item_embeddings, item_id_list = compute_item_embeddings(gallery_embeddings, gallery_paths)
        cluster_to_item_ids = cluster_items(item_embeddings, item_id_list, n_clusters=28)
        sampler = ClusterAwarePKSampler(train_dataset.flat_labels, P=17, K=7, cluster_to_item_ids=cluster_to_item_ids, item_id_to_label=item_id_to_label)
        dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)
        print(f"🔄 Re-clustered at epoch {epoch+1}")

# Evaluation
def evaluate_against_gallery(test_embeddings, test_labels, gallery_embeddings, gallery_labels, test_paths, gallery_paths, top_k_values=[1, 3, 5, 10]):
    print("\n🔍 Evaluating Test Queries Against Full Gallery (excluding self)...")
    similarity = cosine_similarity(test_embeddings, gallery_embeddings)
    correct_at_k = {k: 0 for k in top_k_values}

    for i in range(len(test_embeddings)):
        query_label = test_labels[i]
        query_path = test_paths[i]
        sorted_indices = np.argsort(similarity[i])[::-1]
        filtered_indices = [j for j in sorted_indices if gallery_paths[j] != query_path]
        for k in top_k_values:
            top_k_indices = filtered_indices[:k]
            top_k_labels = gallery_labels[top_k_indices]
            if query_label in top_k_labels:
                correct_at_k[k] += 1

    print("\n📊 Retrieval Accuracy:")
    for k in top_k_values:
        acc = correct_at_k[k] / len(test_embeddings) * 100
        print(f"✅ Top-{k} Accuracy: {acc:.2f}%")

print("\n🚀 Final Evaluation:")
gallery_embeddings, gallery_labels, gallery_paths = extract_embeddings_with_paths(model, gallery_loader)
test_embeddings, test_labels, test_paths = extract_embeddings_with_paths(model, test_loader)
evaluate_against_gallery(test_embeddings, test_labels, gallery_embeddings, gallery_labels, test_paths, gallery_paths)

sil_score = silhouette_score(gallery_embeddings, gallery_labels)
print(f"\n🔍 Silhouette Score: {sil_score:.4f}")
