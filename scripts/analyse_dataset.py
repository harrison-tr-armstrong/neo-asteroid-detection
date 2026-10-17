from pathlib import Path

p = Path(__file__).resolve().parent.parent

datasets_root = (
    p
    / "NEOBench"
    / "NEOBench_data"
    / "datasets"
)

#Find all five dataset folders and print their names and whether they contain train, validation, and test splits
for child in datasets_root.iterdir():
   if child.is_dir():
      print(f"Dataset: {child.name}")

      if (child / "train").is_dir() and (child / "validation").is_dir() and (child / "test").is_dir():


#If the dataset folder contains a train, validation, and test split, print the number of total images
         image_extensions = [".jpg", ".jpeg", ".png"]

         if child.is_dir():
            if (child / "train" / "images").is_dir() and (child / "validation" / "images").is_dir() and (child / "test" / "images").is_dir():
               train_images = [file for file in (child / "train" / "images").rglob("*") if file.is_file() and file.suffix.lower() in image_extensions]
               validation_images = [file for file in (child / "validation" / "images").rglob("*") if file.is_file() and file.suffix.lower() in image_extensions]
               test_images = [file for file in (child / "test" / "images").rglob("*") if file.is_file() and file.suffix.lower() in image_extensions]

               total_images = len(train_images) + len(validation_images) + len(test_images)
               print(f"Train images: {len(train_images)}\nValidation images: {len(validation_images)}\nTest images: {len(test_images)}")
               print(f"Total images: {total_images}")
