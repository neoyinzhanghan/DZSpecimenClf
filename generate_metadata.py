import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def generate_metadata(directories, output_path, train_ratio=0.8):
    data = []
    classes = sorted(directories.keys())  # Alphabetically sorted keys
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    # Iterate over each class and corresponding path
    for cls, path in directories.items():
        class_index = class_indices[cls]
        # Walk through the directory to list all ndpi files
        for root, _, files in tqdm(os.walk(path), desc=f"Processing {cls}"):
            for file in files:
                if file.endswith(".ndpi"):
                    ndpi_path = os.path.join(root, file)
                    data.append((ndpi_path, cls, class_index))

    # Create DataFrame
    df = pd.DataFrame(data, columns=["ndpi_path", "class", "class_index"])

    # Split the data into train and val sets
    train_df, val_df = train_test_split(
        df, test_size=1 - train_ratio, stratify=df["class"]
    )

    # Assign splits
    df["split"] = "val"
    df.loc[train_df.index, "split"] = "train"

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to {output_path}")


# Directory structure
directories = {"BMA_AML": "/media/hdd1/neo/BMA_AML", "PBS": "/media/hdd3/neo/PB_slides"}

# Output path
output_path = os.path.expanduser("~/Documents/neo/wsi_specimen_clf_metadata.csv")

# Generate metadata
generate_metadata(directories, output_path)
