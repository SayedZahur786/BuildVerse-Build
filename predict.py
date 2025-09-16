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
    
    # Create temp_images directory if it doesn't exist
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    save_path = os.path.join(temp_dir, f"temp_{random.randint(1000, 9999)}.jpg")

    try:
        # Get annotated image as NumPy array
        result_array = results.plot()  # BGR format

        # Convert BGR to RGB and save as PIL image
        result_image = Image.fromarray(result_array[..., ::-1])
        
        # Save the image
        result_image.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error saving temp image: {str(e)}")
        return None

# Load YOLO model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "epoch20pt.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

try:
    model = YOLO(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

def predict(image_path, conf_threshold=0.25, top_k=3):
    try:
        # Ensure image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
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
    except Exception as e:
        # Return empty results if prediction fails
        print(f"Prediction failed: {str(e)}")
        return [], None

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
