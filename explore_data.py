import os

data_dir = "data/PlantVillage"

# Confirm karo kya nested PlantVillage folder hai
if os.path.exists(os.path.join(data_dir, "PlantVillage")):
    print("Nested PlantVillage folder mili — is se images empty ho sakti hain, check karte hain")

print(f"\n{'Class':<50} {'Image Count':<10}")
print("-" * 60)

tomato_classes = []
total_images = 0

for folder in sorted(os.listdir(data_dir)):
    full_path = os.path.join(data_dir, folder)
    if os.path.isdir(full_path) and folder.lower().startswith("tomato"):
        images = [f for f in os.listdir(full_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        count = len(images)
        print(f"{folder:<50} {count:<10}")
        tomato_classes.append(folder)
        total_images += count

print("-" * 60)
print(f"Total tomato classes: {len(tomato_classes)}")
print(f"Total tomato images: {total_images}")