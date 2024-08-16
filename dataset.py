import os
import torch
import openslide
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from search_view_indexible import SearchViewIndexible
from torch.utils.data import DataLoader


def ndpi_to_data(ndpi_path):
    """The input is a path to an .ndpi file. The output is tuple of topview_image, search_view_indexible.
    topview_image is a PIL image of the level 7 of the .ndpi file. it is saved in the same directory as the .ndpi file with the same name and .jpg extension.
    search_view_indexible is a SearchViewIndexible object.
    """

    try:
        # Load the .ndpi file
        slide = openslide.OpenSlide(ndpi_path)
    except openslide.OpenSlideError as e:
        print(f"Error loading {ndpi_path}: {e}")
        raise e

    # Get the dimensions of the top view
    top_view_level = 7
    top_view_height, top_view_width = slide.level_dimensions[top_view_level]

    # Extract the top view image
    top_view_image_path = ndpi_path.replace(".ndpi", ".jpg")

    # Check if the top view image already exists
    if not os.path.exists(top_view_image_path):
        top_view_image = slide.read_region(
            (0, 0), top_view_level, (top_view_width, top_view_height)
        )
        top_view_image = top_view_image.convert("RGB")
        top_view_image.save(top_view_image_path, "JPEG", quality=95)
    else:
        top_view_image = Image.open(top_view_image_path)

    # convert to RGB if it is not
    if top_view_image.mode != "RGB":
        top_view_image = top_view_image.convert("RGB")

    # Create a SearchViewIndexible object
    search_view_indexible = SearchViewIndexible(ndpi_path)

    return top_view_image, search_view_indexible


default_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)


class NDPI_Dataset(Dataset):
    def __init__(self, metadata_file, split=None, transforms=default_transform):
        """
        Args:
            metadata_file (string): Path to the csv file with annotations.
            split (string): One of {'train', 'val', 'test'}, which part of the dataset to load.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(metadata_file)
        if split:
            self.metadata = self.metadata[self.metadata["split"] == split]
        self.transforms = transforms
        self.class_weights = self._compute_class_weights()

    def _compute_class_weights(self):
        class_counts = self.metadata["class_index"].value_counts().to_dict()
        total_samples = len(self.metadata)
        class_weights = {
            cls: total_samples / count for cls, count in class_counts.items()
        }
        sample_weights = [
            class_weights[class_idx] for class_idx in self.metadata["class_index"]
        ]
        return torch.DoubleTensor(sample_weights)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ndpi_path = self.metadata.iloc[idx]["ndpi_path"]
        class_index = self.metadata.iloc[idx]["class_index"]
        top_view_image, search_view_indexible = ndpi_to_data(ndpi_path)

        if self.transforms:
            top_view_image = self.transforms(top_view_image)

        return top_view_image, search_view_indexible, class_index


def custom_collate_fn(batch):
    """Custom collate function to handle different data types within a single batch.
    Args:
        batch (list): A list of tuples with (top_view_image, search_view_indexible, class_index).

    Returns:
        tuple: Contains batched images, list of indexibles, and batched class indices.
    """
    # Separate the tuple components into individual lists
    top_view_images = [item[0] for item in batch]
    search_view_indexibles = [item[1] for item in batch]
    class_indices = [item[2] for item in batch]

    # Stack the images and class indices into tensors
    top_view_images = torch.stack(top_view_images, dim=0)
    class_indices = torch.tensor(class_indices, dtype=torch.long)

    # search_view_indexibles remain as a list
    return top_view_images, search_view_indexibles, class_indices


# now write a lightning data module based on the metadata file
# and the custom collate function
class NDPI_DataModule(pl.LightningDataModule):
    def __init__(self, metadata_file, batch_size=32, num_workers=4):
        super().__init__()
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assuming you have a column 'split' in your CSV that contains 'train'/'val' labels
        if stage in (None, "fit"):
            self.train_dataset = NDPI_Dataset(
                self.metadata_file, split="train", transforms=self.transform
            )
            self.val_dataset = NDPI_Dataset(
                self.metadata_file, split="val", transforms=self.transform
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = NDPI_Dataset(
                self.metadata_file, split="val", transforms=self.transform
            )

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            self.train_dataset.class_weights, len(self.train_dataset)
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,  # Use sampler instead of shuffle for balanced sampling
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )
