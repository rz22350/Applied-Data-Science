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
import faiss

# --- Configs ---
SEED = 42
DATASET_PATH = "/user/home/zt22740/ads/uob_image_set/uob_image_set/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECLUSTER_EVERY = 1
N_CLUSTERS = 25
P, K = 16, 4
EMBEDDING_DIM = 256
MARGIN = 1.0
EPOCHS = 30
STEPS_PER_EPOCH = 100
LR = 1e-4
FREEZE_UNTIL = 139  # Freeze all but last 7 layers

# --- Reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Dataset ---
class FashionEvalDataset(Dataset):
    def __init__(self, items, item_ids, transform=None, label_to_index=None):
        self.transform = transform
        self.flat_images = []
        self.flat_labels = []
        for images, item_id in zip(items, item_ids):
            for img_path in images:
                self.flat_images.append(img_path)
                self.flat_labels.append(label_to_index[item_id] if label_to_index else item_id)

    def __len__(self):
        return len(self.flat_images)

    def __getitem__(self, idx):
        image = Image.open(self.flat_images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.flat_labels[idx]

# --- Cluster-Aware PK Sampler ---
class ClusterAwarePKSampler(Sampler):
    def __init__(self, labels, P, K, cluster_to_item_ids, item_id_to_label, seed=42):
        self.P, self.K = P, K
        self.label_to_indices = defaultdict(list)
        self.cluster_to_labels = defaultdict(list)
        self.rng = random.Random(seed)

        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        for cluster_id, item_ids in cluster_to_item_ids.items():
            for item_id in item_ids:
                if item_id in item_id_to_label:
                    label = item_id_to_label[item_id]
                    if label in self.label_to_indices:
                        self.cluster_to_labels[cluster_id].append(label)

    def __iter__(self):
        while True:
            cluster = self.rng.choice(list(self.cluster_to_labels.keys()))
            candidate_labels = self.cluster_to_labels[cluster]
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

# --- Model ---
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            param.requires_grad = i >= FREEZE_UNTIL
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)

# --- Triplet Loss ---
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

# --- Utilities ---
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
    for emb, path in zip(gallery_embeddings, gallery_paths):
        item_id = os.path.basename(os.path.dirname(path))
        item_to_embeddings[item_id].append(emb)
    item_embeddings, item_id_list = [], []
    for item_id, emb_list in item_to_embeddings.items():
        item_embeddings.append(np.mean(emb_list, axis=0))
        item_id_list.append(item_id)
    return np.array(item_embeddings), item_id_list

def cluster_items(item_embeddings, item_ids, n_clusters):
    item_embeddings = np.ascontiguousarray(item_embeddings.astype("float32"))
    res = faiss.StandardGpuResources()
    clustering = faiss.Clustering(item_embeddings.shape[1], n_clusters)
    clustering.niter = 20
    clustering.max_points_per_centroid = 10000000
    index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(item_embeddings.shape[1]))
    clustering.train(item_embeddings, index)
    _, cluster_labels = index.search(item_embeddings, 1)
    cluster_labels = cluster_labels.flatten()
    cluster_to_item_ids = defaultdict(list)
    for item_id, cluster_label in zip(item_ids, cluster_labels):
        cluster_to_item_ids[cluster_label].append(item_id)
    return cluster_to_item_ids

def update_sampler(model, gallery_loader, train_dataset, item_id_to_label):
    gallery_embeddings, _, gallery_paths = extract_embeddings_with_paths(model, gallery_loader)
    item_embeddings, item_id_list = compute_item_embeddings(gallery_embeddings, gallery_paths)
    clusters = cluster_items(item_embeddings, item_id_list, n_clusters=N_CLUSTERS)
    return ClusterAwarePKSampler(train_dataset.flat_labels, P, K, clusters, item_id_to_label)

# --- Dataset Split ---
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

# --- Main Execution ---
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
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = FashionEvalDataset(train_items, train_ids, transform=transform_train, label_to_index=label_to_index)
val_dataset = FashionEvalDataset(val_items, val_ids, transform=transform_eval, label_to_index=label_to_index)
gallery_dataset = FashionEvalDataset(train_items + val_items + test_items, train_ids + val_ids + test_ids, transform=transform_eval, label_to_index=label_to_index)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=4)

model = EmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
sampler = update_sampler(model, gallery_loader, train_dataset, item_id_to_label)
dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(dataloader), total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}")
    for batch_idx, (images, labels) in loop:
        if batch_idx >= STEPS_PER_EPOCH: break
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss = batch_hard_triplet_loss(embeddings, labels, margin=MARGIN)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / STEPS_PER_EPOCH:.4f}")

    if (epoch + 1) % RECLUSTER_EVERY == 0:
        print(f"🔄 Re-clustering at epoch {epoch+1}")
        sampler = update_sampler(model, gallery_loader, train_dataset, item_id_to_label)
        dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)

# --- Evaluation ---
def evaluate_against_gallery(query_embeddings, query_labels, query_paths,
                             gallery_embeddings, gallery_labels, gallery_paths,
                             top_k_values=[1, 3, 5, 10]):
    print("\n🔍 Evaluating Test Queries Against Full Gallery (excluding self)...")
    similarity = cosine_similarity(query_embeddings, gallery_embeddings)
    correct_at_k = {k: 0 for k in top_k_values}

    for i in range(len(query_embeddings)):
        query_label = query_labels[i]
        query_path = query_paths[i]
        sorted_indices = np.argsort(similarity[i])[::-1]
        filtered_indices = [j for j in sorted_indices if gallery_paths[j] != query_path]
        for k in top_k_values:
            top_k_indices = filtered_indices[:k]
            top_k_labels = gallery_labels[top_k_indices]
            if query_label in top_k_labels:
                correct_at_k[k] += 1

    print("\n📊 Retrieval Accuracy:")
    for k in top_k_values:
        acc = correct_at_k[k] / len(query_embeddings) * 100
        print(f"✅ Top-{k} Accuracy: {acc:.2f}%")

# Run final evaluation
test_dataset = FashionEvalDataset(test_items, test_ids, transform=transform_eval, label_to_index=label_to_index)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print("\n🚀 Final Evaluation:")
test_embeddings, test_labels, test_paths = extract_embeddings_with_paths(model, test_loader)
gallery_embeddings, gallery_labels, gallery_paths = extract_embeddings_with_paths(model, gallery_loader)
evaluate_against_gallery(test_embeddings, test_labels, test_paths,
                         gallery_embeddings, gallery_labels, gallery_paths)

torch.save(model.state_dict(), "cluster_triplet_model_weights.pth")
np.save("cluster_gallery_embeddings.npy", gallery_embeddings)
np.save("cluster_gallery_paths.npy", gallery_paths)
