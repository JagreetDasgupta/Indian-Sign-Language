import numpy as np
import cv2
import os
from image_processing import func

# Create directories if they do not exist
os.makedirs("Source Code/data2/train", exist_ok=True)
os.makedirs("Source Code/data2/test", exist_ok=True)

path = "data/train"
path1 = "Source Code/data2"

total_images_processed = 0
images_saved_train = 0
images_saved_test = 0

for dirpath, dirnames, filenames in os.walk(path):
    for dirname in dirnames:
        print(f"Processing class: {dirname}")

        train_dir = os.path.join(path1, "train", dirname)
        test_dir = os.path.join(path1, "test", dirname)

        # Create directories if they do not exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        files = [f for f in os.listdir(os.path.join(path, dirname)) if os.path.isfile(os.path.join(path, dirname, f))]
        num_files = len(files)
        num_train = int(0.75 * num_files)  # 75% for training
        num_test = num_files - num_train    # Remaining for testing

        for idx, file in enumerate(files):
            actual_path = os.path.join(path, dirname, file)
            actual_path1 = os.path.join(train_dir, file)
            actual_path2 = os.path.join(test_dir, file)

            # Load image in grayscale
            img = cv2.imread(actual_path, cv2.IMREAD_GRAYSCALE)

            # Process image using the custom function
            bw_image = func(actual_path)

            # Save image to train or test set based on index
            if idx < num_train:
                images_saved_train += 1
                cv2.imwrite(actual_path1, bw_image)
            else:
                images_saved_test += 1
                cv2.imwrite(actual_path2, bw_image)

            total_images_processed += 1

print("Total images processed:", total_images_processed)
print("Images saved to train set:", images_saved_train)
print("Images saved to test set:", images_saved_test)