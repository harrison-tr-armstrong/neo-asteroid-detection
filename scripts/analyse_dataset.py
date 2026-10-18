from pathlib import Path

#Resolve the project root and locate the NEOBench dataset dir
p = Path(__file__).resolve().parent.parent

#Path to the dir containing all five YOLO datasets
datasets_root = (
    p
    / "NEOBench"
    / "NEOBench_data"
    / "datasets"
)


###############################################
# Count image samples across each dataset split

#Supported image file extensions
image_extensions = [".jpg", ".jpeg", ".png"]

#iterate through each dataset dir
for child in datasets_root.iterdir():
   if child.is_dir():
      print(f"Dataset: {child.name}")

      #Check that the dataset contains all three splits
      if (child / "train").is_dir() and (child / "validation").is_dir() and (child / "test").is_dir():

         #Check that each split has an image dir
         if (child / "train" / "images").is_dir() and (child / "validation" / "images").is_dir() and (child / "test" / "images").is_dir():

            #Collect image files from each split, including nested dir
            train_images = [
               file for file in (child / "train" / "images").rglob("*")
                  if file.is_file() and file.suffix.lower() in image_extensions
               ]
            validation_images = [
               file for file in (child / "validation" / "images").rglob("*")
                  if file.is_file() and file.suffix.lower() in image_extensions
               ]
            test_images = [
               file for file in (child / "test" / "images").rglob("*")
                  if file.is_file() and file.suffix.lower() in image_extensions
               ]

            #Calculate and display the image totals for each split and the full dataset
            total_images = len(train_images) + len(validation_images) + len(test_images)
            print(f"Train images: {len(train_images)}\nValidation images: {len(validation_images)}\nTest images: {len(test_images)}")
            print(f"Total images: {total_images}")


################################################
# Count distinct class IDs from YOLO annotations

#Supported YOLO label file extension
label_extensions = [".txt"]

for child in datasets_root.iterdir():
   if child.is_dir():
      print(f"Dataset: {child.name}")

      if (child/ "train").is_dir() and (child / "validation").is_dir() and (child / "test").is_dir():

         #Store unique class IDs found across all three splits
         class_ids = set()

         if (child / "train" / "labels").is_dir() and (child / "validation" / "labels").is_dir() and (child / "test" / "labels").is_dir():

            #Find all label files in the training split
            train_labels = [
               file for file in (child / "train" / "labels").rglob("*")
                  if file.is_file() and file.suffix.lower() in label_extensions
            ]
            #Read each annotation file and extract its class IDs
            for file in train_labels:
               with open(file, "r") as f:
                  for line in f:
                     #Split the annotation line into its YOLO values
                     value = line.split()

                     #Skip blank or whitespace lines
                     if not value:
                        continue

                     #Add the class ID to the set, avoiding duplicates
                     class_ids.add(int(value[0]))

            validation_labels = [
               file for file in (child / "validation" / "labels").rglob("*")
                  if file.is_file() and file.suffix.lower() in label_extensions
            ]
            for file in validation_labels:
               with open(file, "r") as f:
                  for line in f:
                     value = line.split()
                     if not value:
                        continue

                     class_ids.add(int(value[0]))

            test_labels = [
               file for file in (child / "test" / "labels").rglob("*")
                  if file.is_file() and file.suffix.lower() in label_extensions
            ]
            for file in test_labels:
               with open(file, "r") as f:
                  for line in f:
                     value = line.split()
                     if not value:
                        continue

                     class_ids.add(int(value[0]))

            #Display the distinct classes found across all three splits
            print(f"Distinct classes across all splits: {len(class_ids)}")
            print(f"Total classes: {sorted(class_ids)}")
