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

    # Store the gradients (analytical gradients)
    backward_gradients = [param.grad.clone() for param in model.parameters()]

    return backward_gradients

def compute_numerical_gradient(model, input_data, target_data, loss_fn, epsilon=1e-5):
    # Store numerical gradients for all parameters
    numerical_gradients = []

    # Get model parameters
    params = list(model.parameters())

    # Forward pass without gradients
    model.eval()
    topview_image, search_view_indexible = input_data
    topview_image.requires_grad = False  # Disable gradient tracking

    for param in params:
        num_grad = torch.zeros_like(param)
        param_data = param.data

        # Perturb each element of the parameter tensor
        for i in range(param_data.numel()):
            param_data_flat = param_data.view(-1)  # Flatten parameter tensor
            orig = param_data_flat[i].item()

            # Perturb with +epsilon
            param_data_flat[i] = orig + epsilon
            output_plus = model(topview_image, search_view_indexible)
            loss_plus = loss_fn(output_plus, target_data)

            # Perturb with -epsilon
            param_data_flat[i] = orig - epsilon
            output_minus = model(topview_image, search_view_indexible)
            loss_minus = loss_fn(output_minus, target_data)

            # Reset to original value
            param_data_flat[i] = orig

            # Compute numerical gradient
            grad_approx = (loss_plus.item() - loss_minus.item()) / (2 * epsilon)
            num_grad.view(-1)[i] = grad_approx

        numerical_gradients.append(num_grad)

    return numerical_gradients


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
    from DZSpecimenClfToy import DZSpecimenClfToy
    import torch.nn as nn

    class SpecimenClassifier(nn.Module):
        def __init__(self, N, num_classes=2, patch_size=224):
            super(SpecimenClassifier, self).__init__()
            self.model = DZSpecimenClfToy(
                N, num_classes=num_classes, patch_size=patch_size
            )

        def forward(self, topview_image_tensor, search_view_indexibles):
            return self.model(topview_image_tensor, search_view_indexibles)

    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 1
    N = 1  # Example value
    patch_size = 4
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

    print("Computing backward (analytical) gradient...")

    # Compute backward gradients (analytical)
    backward_gradients = compute_backward_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    print("Saving full backward gradients to JSON...")

    # Save the full backward gradients to a JSON file
    save_gradients_to_json(backward_gradients)

    print("Saving backward gradient ranges to JSON...")

    # Save the min/max range of backward gradients to a JSON file
    save_gradient_range_to_json(backward_gradients)

    print("Computing numerical gradient...")

    # Compute numerical gradients
    numerical_gradients = compute_numerical_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    print("Saving numerical gradients to JSON...")

    # Save the numerical gradients to a JSON file
    save_gradients_to_json(numerical_gradients, file_name="numerical_gradients_full.json")

    print("Saving numerical gradient ranges to JSON...")

    # Save the min/max range of numerical gradients to a JSON file
    save_gradient_range_to_json(numerical_gradients, file_name="numerical_gradients_range.json")
