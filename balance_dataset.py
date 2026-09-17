import os
import random
from PIL import Image, ImageEnhance
import shutil

random.seed(42)  # reproducibility ke liye

source_dir = "data/PlantVillage"
output_dir = "data/tomato_balanced"
target_count = 1500

os.makedirs(output_dir, exist_ok=True)

def augment_image(img):
    """Ek chhota, random augmentation apply karta hai"""
    choice = random.choice(["rotate", "flip", "brightness"])
    if choice == "rotate":
        angle = random.choice([90, 180, 270])
        return img.rotate(angle)
    elif choice == "flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif choice == "brightness":
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)

tomato_folders = [f for f in os.listdir(source_dir) if f.lower().startswith("tomato") and os.path.isdir(os.path.join(source_dir, f))]

for folder in sorted(tomato_folders):
    src_path = os.path.join(source_dir, folder)
    dst_path = os.path.join(output_dir, folder)
    os.makedirs(dst_path, exist_ok=True)

    images = [f for f in os.listdir(src_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    count = len(images)

    print(f"{folder}: {count} original images -> target {target_count}")

    if count >= target_count:
        # Undersample: randomly target_count images select karo
        selected = random.sample(images, target_count)
        for img_name in selected:
            shutil.copy(os.path.join(src_path, img_name), os.path.join(dst_path, img_name))
    else:
        # Pehle saari original images copy karo
        for img_name in images:
            shutil.copy(os.path.join(src_path, img_name), os.path.join(dst_path, img_name))

        # Phir augmented images bana kar target tak pahuncho
        needed = target_count - count
        for i in range(needed):
            src_img_name = random.choice(images)
            img = Image.open(os.path.join(src_path, src_img_name)).convert("RGB")
            augmented = augment_image(img)
            augmented.save(os.path.join(dst_path, f"aug_{i}_{src_img_name}"))

print("\nBalancing complete!")