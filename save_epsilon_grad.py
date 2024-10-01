import torch
import numpy as np
import random
import csv


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


def compute_backward_gradient(model, input_data, target_data, loss_fn):
    # Perform forward pass and compute loss
    topview_image, search_view_indexible = input_data
    output = model(topview_image, search_view_indexible)
    loss = loss_fn(output, target_data)

    # Zero out any existing gradients
    model.zero_grad()

    # Compute gradients using backpropagation
    loss.backward()

    # Extract gradients from the model parameters
    backward_gradients = [param.grad.clone() for param in model.parameters()]
    return backward_gradients


def compare_gradients(numerical_gradients, backward_gradients, csv_filename="grad_comparison.csv"):
    device = "cpu"
    numerical_gradients = [grad.to(device) for grad in numerical_gradients]
    backward_gradients = [grad.to(device) for grad in backward_gradients]

    # Flatten all gradients for comparison
    numerical_grad_flat = np.concatenate([grad.flatten().cpu().numpy() for grad in numerical_gradients])
    backward_grad_flat = np.concatenate([grad.flatten().cpu().numpy() for grad in backward_gradients])

    # Compare gradients
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter Index", "Numerical Gradient", "Backward Gradient", "Relative Error"])

        for idx, (num_grad, back_grad) in enumerate(zip(numerical_grad_flat, backward_grad_flat)):
            relative_error = np.linalg.norm(back_grad - num_grad) / (np.linalg.norm(back_grad) + np.linalg.norm(num_grad) + 1e-8)
            writer.writerow([idx, num_grad, back_grad, relative_error])

    print(f"Gradient comparison saved to {csv_filename}")


if __name__ == "__main__":
    from dataset import NDPI_DataModule
    from DZSpecimenClfToy import DZSpecimenClfToy
    import torch.nn as nn

    class SpecimenClassifier(nn.Module):
        def __init__(self, N, num_classes=2, patch_size=224):
            super(SpecimenClassifier, self).__init__()
            self.model = DZSpecimenClfToy(N, num_classes=num_classes, patch_size=patch_size)

        def forward(self, topview_image_tensor, search_view_indexibles):
            return self.model(topview_image_tensor, search_view_indexibles)

    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 1
    N = 1
    patch_size = 4
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module = NDPI_DataModule(metadata_file, batch_size, num_workers=64)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    model = SpecimenClassifier(N, patch_size=patch_size, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    batch = next(iter(train_loader))
    topview_image, search_view_indexible, class_index = batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topview_image = topview_image.to(device)
    class_index = class_index.to(device)

    print("Computing numerical gradient...")

    model.to("cpu")
    topview_image = topview_image.to("cpu")
    class_index = class_index.to("cpu")

    # Compute numerical gradients
    numerical_gradients = compute_numerical_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    print("Computing backward gradient...")

    model.to(device)
    topview_image = topview_image.to(device)
    class_index = class_index.to(device)

    backward_gradients = compute_backward_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    print("Comparing gradients...")

    # Compare and save gradients to CSV
    compare_gradients(numerical_gradients, backward_gradients, "grad_comparison.csv")
