import torch
import torch.nn as nn
import time
from DZSpecimenClfToy import DZSpecimenClfToy
from dataset import NDPI_DataModule


class SpecimenClassifier(nn.Module):
    def __init__(self, N, num_classes=2, patch_size=224):
        super(SpecimenClassifier, self).__init__()
        self.model = DZSpecimenClfToy(N, num_classes=num_classes, patch_size=patch_size)

    def forward(self, topview_image_tensor, search_view_indexibles):
        return self.model(topview_image_tensor, search_view_indexibles)


def measure_time(device, model, topview_image, search_view_indexible):
    model.to(device)
    topview_image = topview_image.to(device)

    # Make sure GPU is ready before starting time measurement
    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        outputs = model(topview_image, search_view_indexible)

    # Make sure all GPU operations finish before stopping the timer
    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    end_time = time.time()

    return end_time - start_time


def main():
    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 1
    N = 1  # Example value
    patch_size = 4
    num_classes = 2  # Number of classes in your dataset

    # Instantiate dataset and dataloader
    data_module = NDPI_DataModule(metadata_file, batch_size, num_workers=64)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Get one batch of data
    topview_image, search_view_indexible, class_index = next(iter(train_loader))

    # Instantiate model
    model = SpecimenClassifier(N, patch_size=patch_size, num_classes=num_classes)

    # Measure time on CPU
    cpu_time = measure_time(
        torch.device("cpu"), model, topview_image, search_view_indexible
    )
    print(f"Forward pass time on CPU: {cpu_time:.6f} seconds")

    # Measure time on GPU (if available)
    if torch.cuda.is_available():
        gpu_time = measure_time(
            torch.device("cuda"), model, topview_image, search_view_indexible
        )
        print(f"Forward pass time on GPU: {gpu_time:.6f} seconds")
    else:
        print("GPU is not available.")

    # Output:
    # print the total number of trainable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")


if __name__ == "__main__":
    main()
