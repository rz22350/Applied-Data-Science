import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import ResNet50_Weights

# ✅ Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ Best Hyperparameters from Optuna
BEST_HYPERPARAMS = {
    'embedding_dim': 512,
    'learning_rate': 1.4824907432786402e-05,
    'margin': 1.5694574740347167,
    'weight_decay': 2.4823534471783044e-05,
    'batch_size': 32,
    'unfreeze_layers': 7
}

# Print the best hyperparameters being used
print("\n✅ Best Hyperparameters Applied:")
for key, value in BEST_HYPERPARAMS.items():
    print(f"   🔹 {key}: {value}")

# 📂 Step 1: Set Dataset Path
DATASET_PATH = "/user/home/zt22740/ads/uob_image_set/uob_image_set/"
SEED = 42  # Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior

print(f"✅ Random seed set to {SEED}")
# 🏷 Step 2: Define Dataset Classes
class FashionTripletDataset(Dataset):
    """Dataset that returns triplets for training with triplet loss"""
    def __init__(self, items, item_ids, transform=None):
        self.items = items
        self.item_ids = item_ids
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        images = self.items[idx]
        item_id = self.item_ids[idx]

        anchor_path = random.choice(images)
        positive_path = random.choice(images)

        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.items) - 1)
        negative_images = self.items[negative_idx]
        negative_path = random.choice(negative_images)

        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, item_id

class FashionEvalDataset(Dataset):
    """Dataset for evaluation"""
    def __init__(self, items, item_ids, transform=None):
        self.transform = transform
        self.flat_images = []
        self.flat_labels = []

        for images, item_id in zip(items, item_ids):
            for img_path in images:
                self.flat_images.append(img_path)
                self.flat_labels.append(item_id)

    def __len__(self):
        return len(self.flat_images)

    def __getitem__(self, idx):
        image_path = self.flat_images[idx]
        label = self.flat_labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# 🎨 Step 3: Data Preprocessing with Enhanced Augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Reduced jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 📂 Step 4: Load and Split Dataset
def load_and_split_dataset(dataset_path):
    all_items = []
    item_ids = []

    for clothing_item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, clothing_item)
        if os.path.isdir(item_path):
            images = [os.path.join(item_path, img) for img in sorted(os.listdir(item_path)) if img.endswith(".jpg")]
            if len(images) >= 2:
                all_items.append(images)
                item_ids.append(clothing_item)

    train_size = int(0.8 * len(all_items))
    val_size = int(0.1 * len(all_items))

    train_items = all_items[:train_size]
    val_items = all_items[train_size:train_size+val_size]
    test_items = all_items[train_size+val_size:]

    train_ids = item_ids[:train_size]
    val_ids = item_ids[train_size:train_size+val_size]
    test_ids = item_ids[train_size+val_size:]

    return train_items, val_items, test_items, train_ids, val_ids, test_ids

# 🔀 Step 5: Create Datasets and DataLoaders
train_items, val_items, test_items, train_ids, val_ids, test_ids = load_and_split_dataset(DATASET_PATH)

train_dataset = FashionTripletDataset(train_items, train_ids, transform=transform_train)
val_dataset = FashionEvalDataset(val_items, val_ids, transform=transform_eval)
test_dataset = FashionEvalDataset(test_items, test_ids, transform=transform_eval)

train_dataloader = DataLoader(train_dataset, batch_size=BEST_HYPERPARAMS['batch_size'], shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BEST_HYPERPARAMS['batch_size'], shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=BEST_HYPERPARAMS['batch_size'], shuffle=False, num_workers=4)

# 🧠 Step 6: Define Model with Unfreezing Layers
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=BEST_HYPERPARAMS['embedding_dim']):
        super(EmbeddingNet, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Unfreeze the last 'unfreeze_layers' layers
        for param in list(self.backbone.parameters())[-BEST_HYPERPARAMS['unfreeze_layers']:]:
            param.requires_grad = True

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

        self.l2_norm = lambda x: x / torch.norm(x, p=2, dim=1, keepdim=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.l2_norm(x)
        return x

# 🔥 Step 7: Define Loss & Optimizer
model = EmbeddingNet().to(device)
criterion = nn.TripletMarginLoss(margin=BEST_HYPERPARAMS['margin'])
optimizer = optim.Adam(model.parameters(), lr=BEST_HYPERPARAMS['learning_rate'], weight_decay=BEST_HYPERPARAMS['weight_decay'])

# 📊 Step 8: Evaluate the Model (Top-K Retrieval)
def evaluate_model(model, test_dataloader):
    print("\n🔍 Final Model Evaluation...")
    model.eval()
    test_embeddings = []
    test_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Extracting test embeddings"):
            images = images.to(device)
            emb = model(images).cpu().numpy()
            test_embeddings.append(emb)
            test_labels.extend(labels)

    test_embeddings = np.vstack(test_embeddings)
    similarity_matrix = cosine_similarity(test_embeddings)

    # Convert test_labels from Tensor to NumPy array for proper indexing
    test_labels = np.array(test_labels)

    # Compute Top-K retrieval accuracy
    top_k_values = [1, 3, 5, 10]
    correct_at_k = {k: 0 for k in top_k_values}

    for i in range(len(test_labels)):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1][1:]  # Get sorted indices excluding self-match

        retrieved_labels = test_labels[sorted_indices]  # Get the actual labels of retrieved items

        for k in top_k_values:
            if test_labels[i] in retrieved_labels[:k]:  # Check if correct label exists in Top-K
                correct_at_k[k] += 1

    for k in top_k_values:
        accuracy = correct_at_k[k] / len(test_labels)
        print(f"✅ Top-{k} Retrieval Accuracy: {accuracy * 100:.2f}%")

# 📊 Step 9: Run Final Evaluation
evaluate_model(model, test_dataloader)
