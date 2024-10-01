import torch
import numpy as np
from scipy.optimize import approx_fprime
import random
import csv

def compute_numerical_gradient(model, input_data, target_data, loss_fn, epsilon=1e-5, n_params=100):
    # Flatten model parameters into a 1D numpy array
    params_flat = np.concatenate(
        [param.detach().cpu().numpy().flatten() for param in model.parameters()]
    )

    # Randomly select n_params indices if specified
    param_indices = np.arange(len(params_flat))
    if n_params:
        param_indices = np.random.choice(param_indices, size=n_params, replace=False)

    # Define the loss function wrapper
    def loss_fn_wrapper(params_flat):
        start_idx = 0
        for param in model.parameters():
            param_shape = param.shape
            numel = param.numel()
            param.data = torch.tensor(
                params_flat[start_idx: start_idx + numel].reshape(param_shape),
                dtype=param.dtype,
            )
            start_idx += numel

        topview_image, search_view_indexible = input_data
        output = model(topview_image, search_view_indexible)
        loss = loss_fn(output, target_data).item()
        return loss

    # Compute numerical gradient using approx_fprime for selected indices
    numerical_gradient_flat = np.zeros_like(params_flat)
    selected_grad = approx_fprime(params_flat[param_indices], loss_fn_wrapper, epsilon)
    numerical_gradient_flat[param_indices] = selected_grad

    # Reshape numerical gradient back to model parameter shapes
    numerical_gradients = []
    start_idx = 0
    for param in model.parameters():
        numel = param.numel()
        grad_shape = param.shape
        numerical_gradients.append(
            torch.tensor(
                numerical_gradient_flat[start_idx: start_idx + numel].reshape(grad_shape)
            )
        )
        start_idx += numel

    return numerical_gradients, param_indices


def compute_backward_gradient(model, input_data, target_data, loss_fn):
    topview_image, search_view_indexible = input_data
    output = model(topview_image, search_view_indexible)
    loss = loss_fn(output, target_data)

    model.zero_grad()
    loss.backward()

    backward_gradients = [param.grad.clone() for param in model.parameters()]
    return backward_gradients


def compare_gradients(numerical_gradients, backward_gradients, param_indices, csv_filename="grad_comparison.csv"):
    device = "cpu"
    numerical_gradients = [grad.to(device) for grad in numerical_gradients]
    backward_gradients = [grad.to(device) for grad in backward_gradients]

    # Compare and save the selected parameter gradients in a CSV
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter Index", "Numerical Gradient", "Backward Gradient", "Relative Error"])

        for idx, param_idx in enumerate(param_indices):
            num_grad = numerical_gradients[idx].flatten()[param_idx]
            back_grad = backward_gradients[idx].flatten()[param_idx]
            relative_error = torch.norm(back_grad - num_grad) / (torch.norm(back_grad) + torch.norm(num_grad) + 1e-8)  # Adding epsilon for numerical stability
            writer.writerow([param_idx, num_grad.item(), back_grad.item(), relative_error.item()])

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

    # Compute numerical gradients for a subset of parameters
    N_params = 100  # Number of randomly selected parameters
    numerical_gradients, selected_indices = compute_numerical_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn, n_params=N_params
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
    compare_gradients(numerical_gradients, backward_gradients, selected_indices, "grad_comparison.csv")
