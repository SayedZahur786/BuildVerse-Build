import os
from ultralytics import YOLO
from PIL import Image

# Define label names (update these if your model's labels are different)
label_names = ["Acne", "Pimples", "Bacterial Breakouts", "Blackheads", "Clogged Pores",
               "Oily Skin", "Open Pores", "Fine Lines", "Wrinkles", "Uneven Texture",
               "Dull Skin", "Hyperpigmentation", "Uneven Skin Tone", "Dark Spots", "PIH",
               "Melasma", "Redness", "Irritation", "PIE", "Rosacea", "Barrier Damage",
               "Sensitive Skin", "Dry Skin", "Dehydration", "Eczema", "Barrier Repair",
               "Puffiness", "Dark Circles", "Sunburn", "Sun Damage"]

def save_temp_img(results):
    import random
    save_path = "temp_images/temp_" + str(random.randint(1000, 9999)) + ".jpg"

    # Get annotated image as NumPy array
    result_array = results.plot()  # BGR format

    # Convert BGR to RGB and save as PIL image
    result_image = Image.fromarray(result_array[..., ::-1])
    
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the image
    result_image.save(save_path)

    return save_path

# Load YOLO model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "epoch20pt.pt")
model = YOLO(model_path)

def predict(image_path, conf_threshold=0.25, top_k=3):
    results = model(image_path, conf=conf_threshold)[0]
    save_path = save_temp_img(results)

    detected = []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = label_names[cls] if cls < len(label_names) else f"Class_{cls}"
        detected.append((label, conf))

    # Sort and get top_k
    top_labels = sorted(detected, key=lambda x: x[1], reverse=True)[:top_k]
    top_concerns = [label for label, _ in top_labels]
    top_concerns = list(set(top_concerns))  # Remove duplicates
    return top_concerns, save_path

# Optional: for testing
if __name__ == "__main__":
    test_image_path = os.path.join(current_dir, "images", "1.jpg")
    if os.path.exists(test_image_path):
        predicted_labels = predict(test_image_path)
        print("\nTop Skin Concerns:")
        if predicted_labels:
            print(", ".join(list(set(predicted_labels))))
        else:
            print("No significant skin concerns detected.")
    else:
        print(f"Test image not found at: {test_image_path}")
