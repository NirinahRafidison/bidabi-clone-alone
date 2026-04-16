import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# -------------------
# Config
# -------------------
DATA_DIR = "data/raw/images"
BATCH_SIZE = 8
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Transformations
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# -------------------
# Dataset
# -------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print("Classes:", dataset.classes)

# -------------------
# Split
# -------------------
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -------------------
# Model
# -------------------
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------
# Tracking metrics
# -------------------
train_losses = []
val_losses = []

best_loss = float("inf")

# -------------------
# Training loop
# -------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Moyenne
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Stockage pour graphiques
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Sauvegarde du meilleur modèle
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        print("✔ Best model saved")

print("Training finished")

# -------------------
# Graphiques
# -------------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.show()