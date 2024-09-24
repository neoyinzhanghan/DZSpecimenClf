import torch
import numpy as np
import json

def compute_backward_gradient(model, input_data, target_data, loss_fn):
    # Perform a forward pass to compute the output and loss
    topview_image, search_view_indexible = input_data
    output = model(topview_image, search_view_indexible)
    loss = loss_fn(output, target_data)

    # Zero the gradients before the backward pass
    model.zero_grad()

    # Perform backward pass to compute gradients
    loss.backward()

    # Store the gradients
    backward_gradients = [param.grad.clone() for param in model.parameters()]

    return backward_gradients


def save_gradients_to_json(gradients, file_name="backward_gradients_full.json"):
    gradients_dict = {}
    for idx, grad in enumerate(gradients):
        gradients_dict[f"Parameter_{idx}"] = grad.flatten().cpu().numpy().tolist()

    with open(file_name, mode="w") as file:
        json.dump(gradients_dict, file, indent=4)


def save_gradient_range_to_json(gradients, file_name="backward_gradients_range.json"):
    gradient_ranges_dict = {}
    for idx, grad in enumerate(gradients):
        min_val = grad.min().item()
        max_val = grad.max().item()
        gradient_ranges_dict[f"Parameter_{idx}"] = {"min": min_val, "max": max_val}

    with open(file_name, mode="w") as file:
        json.dump(gradient_ranges_dict, file, indent=4)


if __name__ == "__main__":
    from dataset import NDPI_DataModule
    from DZSpecimenClf import DZSpecimenClf
    import torch.nn as nn

    class SpecimenClassifier(nn.Module):
        def __init__(self, N, num_classes=2, patch_size=224):
            super(SpecimenClassifier, self).__init__()
            self.model = DZSpecimenClf(
                N, num_classes=num_classes, patch_size=patch_size
            )

        def forward(self, topview_image_tensor, search_view_indexibles):
            return self.model(topview_image_tensor, search_view_indexibles)

    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 1
    N = 1  # Example value
    patch_size = 224
    num_classes = 2  # Number of classes in your dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module = NDPI_DataModule(metadata_file, batch_size, num_workers=64)

    # Instantiate dataset and dataloaders
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Instantiate model, loss, optimizer, and metrics
    model = SpecimenClassifier(N, patch_size=patch_size, num_classes=num_classes).to(
        device
    )
    loss_fn = nn.CrossEntropyLoss()

    # get the first batch of data
    batch = next(iter(train_loader))
    topview_image, search_view_indexible, class_index = batch
    topview_image = topview_image.to(device)
    class_index = class_index.to(device)

    print("Computing backward gradient...")

    # Compute backward gradients
    backward_gradients = compute_backward_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    print("Saving full gradients to JSON...")

    # Save the full gradients to a JSON file
    save_gradients_to_json(backward_gradients)

    print("Saving gradient ranges to JSON...")

    # Save the min/max range of gradients to a JSON file
    save_gradient_range_to_json(backward_gradients)
