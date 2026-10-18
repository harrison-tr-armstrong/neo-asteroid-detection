from pathlib import Path
from dataset_loader import NEODatasetLoader


project_root = Path(__file__).resolve().parent.parent

dataset_directory = (
    project_root
    / "NEOBench"
    / "NEOBench_data"
    / "datasets"
    / "HubbleAsteroidHunter_YOLO"
)

# Select the dataset split: train, validation, or test.
loader = NEODatasetLoader(dataset_directory, split="train")

print(f"Total images: {len(loader)}")

sample = loader.get_sample(126)

print(f"Image: {sample['image_path'].name}")
print(f"Label exists: {sample['label_exists']}")
print(f"Valid annotations: {len(sample['annotations'])}")
print(f"Errors: {sample['errors']}")

loader.show_sample(sample)
