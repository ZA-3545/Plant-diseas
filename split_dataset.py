import os
import shutil
import random

random.seed(42)

source_dir = "data/tomato_balanced"
output_dir = "data/tomato_split"

splits = {"train": 0.70, "val": 0.15, "test": 0.15}

for split_name in splits:
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

classes = sorted(os.listdir(source_dir))

for class_name in classes:
    class_path = os.path.join(source_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * splits["train"])
    n_val = int(n_total * splits["val"])
    # baaki sab test mein (rounding errors handle karne ke liye)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        dst_class_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        for img_name in split_imgs:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(dst_class_dir, img_name))

    print(f"{class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

print("\nSplit complete!")