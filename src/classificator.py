import torch
from torchvision import models, transforms
from PIL import Image
import os

# Classes
classes = ["bread", "milk", "butter"]

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

# -------------------
# MAIN
# -------------------
if __name__ == "__main__":

    # Si aucune image donnée → utiliser une image par défaut
    default_image = "data/raw/images/bread/0.jpg"

    if os.path.exists(default_image):
        result = predict(default_image)
        print("Prediction (default image):", result)
    else:
        print("No image provided and default image not found.")