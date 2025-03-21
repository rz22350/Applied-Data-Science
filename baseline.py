import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ✅ Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset Path
DATASET_PATH = "/user/home/zt22740/ads/uob_image_set/uob_image_set"

# ✅ Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load Dataset
dataset = ImageFolder(root=DATASET_PATH, transform=transform)

# ✅ Store Image Paths for Retrieval
file_paths = [sample[0] for sample in dataset.samples]  # Get filenames

# ✅ Fix: Use integer category indices instead of class names
category_indices = np.arange(len(dataset.classes))  # Get class indices

# ✅ Split Dataset (Train 80%, Val 10%, Test 10%) using category indices
train_categories, temp_categories = train_test_split(category_indices, test_size=0.20, random_state=42)
val_categories, test_categories = train_test_split(temp_categories, test_size=0.50, random_state=42)

# ✅ Assign images to the correct dataset split based on their labels
train_indices = [i for i, (_, label) in enumerate(dataset.samples) if label in train_categories]
val_indices = [i for i, (_, label) in enumerate(dataset.samples) if label in val_categories]
test_indices = [i for i, (_, label) in enumerate(dataset.samples) if label in test_categories]

# ✅ Create dataset subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# ✅ Load Dataset into DataLoader
full_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"✅ Dataset Split: {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")
print(f"✅ Using {len(test_dataset)} test images as queries while comparing against all {len(dataset)} images.")

# 📊 **Feature Extractor using ResNet-50**
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.backbone(x)

# ✅ Load Pretrained Model for Feature Extraction
model = FeatureExtractor().to(device)
model.eval()

# ✅ Extract Features for All Images
embeddings = []
labels = []

with torch.no_grad():
    for images, targets in tqdm(full_loader, desc="Extracting embeddings for all dataset"):
        images = images.to(device)
        emb = model(images).cpu().numpy()
        embeddings.append(emb)
        labels.extend(targets.cpu().numpy())

embeddings = np.vstack(embeddings)

# ✅ Save embeddings, labels, and file paths for retrieval
np.save("embeddings_baseline.npy", embeddings)
np.save("labels_baseline.npy", np.array(labels))
np.save("file_paths.npy", np.array(file_paths))  # Store filenames

print("✅ Embeddings saved for the entire dataset.")

# ✅ Compute Retrieval Accuracy and Display Retrieved Filenames
def compute_top_k_accuracy(embeddings, labels, test_indices, top_k_values=[1, 3, 5, 10]):
    correct_at_k = {k: 0 for k in top_k_values}
    num_queries = len(test_indices)

    similarity_matrix = cosine_similarity(embeddings)  # Compute similarities for all images

    # ✅ Load filenames
    file_paths = np.load("file_paths.npy", allow_pickle=True)

    for i in tqdm(test_indices, desc="Evaluating Top-K Retrieval Accuracy"):
        query_label = labels[i]
        query_filename = file_paths[i]  # Get query image filename
        sorted_indices = np.argsort(similarity_matrix[i])[::-1][1:]  # Exclude self-match

        print(f"\n🔍 Query Image: {query_filename}")  # Print query filename

        for k in top_k_values:
            top_k_indices = sorted_indices[:k]  # Get top-K retrieved indices
            retrieved_filenames = [file_paths[j] for j in top_k_indices]  # Convert indices to filenames

            print(f"📌 Top-{k} Retrieved Images:")
            for retrieved_file in retrieved_filenames:
                print(f"   - {retrieved_file}")  # Print retrieved image filenames

            # ✅ Check if at least one retrieved image has the same category
            if any(labels[j] == query_label for j in top_k_indices):
                correct_at_k[k] += 1

    # ✅ Print Accuracy Results
    for k in top_k_values:
        accuracy = correct_at_k[k] / num_queries * 100
        print(f"✅ Top-{k} Retrieval Accuracy: {accuracy:.2f}%")

# ✅ Run Retrieval and Display Filenames
compute_top_k_accuracy(embeddings, labels, test_indices)

# ✅ Load saved embeddings and labels
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

# ✅ Reduce embeddings to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# ✅ Create a scatter plot of t-SNE results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette="viridis", alpha=0.7)
plt.title("t-SNE Visualization of Image Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
tsne_plot_path = "tsne_embeddings_new.png"
plt.savefig(tsne_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ t-SNE visualization saved as {tsne_plot_path}")
