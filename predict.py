import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import SkinCNN
import os

num_classes = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SkinCNN(num_classes).to(device)

# Get the absolute path to the current directory (predict.py's location)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "skin_model.pth")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

label_names = ["Acne", "Pimples", "Bacterial Breakouts", "Blackheads", "Clogged Pores",
               "Oily Skin", "Open Pores", "Fine Lines", "Wrinkles", "Uneven Texture",
               "Dull Skin", "Hyperpigmentation", "Uneven Skin Tone", "Dark Spots", "PIH",
               "Melasma", "Redness", "Irritation", "PIE", "Rosacea", "Barrier Damage",
               "Sensitive Skin", "Dry Skin", "Dehydration", "Eczema", "Barrier Repair",
               "Puffiness", "Dark Circles", "Sunburn", "Sun Damage"]

def predict(image_path, threshold=0.08, top_k=3):  
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predictions = torch.sigmoid(outputs).cpu().numpy().flatten()

    detected = [(label, score) for label, score in zip(label_names, predictions) if score > threshold]
    top_labels = sorted(detected, key=lambda x: x[1], reverse=True)[:top_k]
    top_concerns = [label for label, _ in top_labels]
    return top_concerns

# Optional: Only runs for standalone testing
if __name__ == "__main__":
    test_image_path = os.path.join(current_dir, "images", "1.jpg")
    if os.path.exists(test_image_path):
        predicted_labels = predict(test_image_path)
        print("\nTop Skin Concerns:")
        if predicted_labels:
            print(", ".join(predicted_labels))
        else:
            print("No significant skin concerns detected.")
    else:
        print(f"Test image not found at: {test_image_path}")
