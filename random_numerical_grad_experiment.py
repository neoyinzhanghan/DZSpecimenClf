import torch
import numpy as np
import csv
import torch.nn as nn
from tqdm import tqdm
from dataset import NDPI_DataModule
from DZSpecimenClf import DZSpecimenClf
from param_flat import flatten_parameters, unflatten_parameters


class SpecimenClassifier(nn.Module):
    def __init__(self, N, num_classes=2, patch_size=224):
        super(SpecimenClassifier, self).__init__()
        self.model = DZSpecimenClf(N, num_classes=num_classes, patch_size=patch_size)

    def forward(self, topview_image_tensor, search_view_indexibles):
        return self.model(topview_image_tensor, search_view_indexibles)


def compute_numerical_gradient(model, input_data, target_data, loss_fn, epsilon=1e-3, n_params=None):
    numerical_gradients = []
    params = list(model.parameters())
    param_shapes = [param.shape for param in params]
    total_params = sum(param.numel() for param in params)

    if n_params:
        param_indices = np.random.choice(total_params, size=n_params, replace=False)
    else:
        param_indices = np.arange(total_params)

    param_list = flatten_parameters(params)

    for idx in tqdm(param_indices, desc="Computing numerical gradients"):
        original_value = param_list[idx].item()

        param_list[idx] = original_value + epsilon
        set_model_params(model, param_list, param_shapes)
        output_plus = model(*input_data)
        loss_plus = loss_fn(output_plus, target_data).item()

        param_list[idx] = original_value - epsilon
        set_model_params(model, param_list, param_shapes)
        output_minus = model(*input_data)
        loss_minus = loss_fn(output_minus, target_data).item()

        param_list[idx] = original_value
        set_model_params(model, param_list, param_shapes)

        grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
        numerical_gradients.append(grad_approx)

    return numerical_gradients, param_indices


def set_model_params(model, flat_params, param_shapes):
    unflattened_params = unflatten_parameters(flat_params, param_shapes)
    with torch.no_grad():
        for param, unflattened_param in zip(model.parameters(), unflattened_params):
            param.copy_(unflattened_param)


def compute_backward_gradient(model, input_data, target_data, loss_fn):
    output = model(*input_data)
    loss = loss_fn(output, target_data)
    model.zero_grad()
    loss.backward()
    backward_gradients = [param.grad.clone() for param in model.parameters()]
    return backward_gradients


def compare_gradients(
    numerical_gradients, backward_gradients, param_indices, csv_filename="eps_grad_comparison.csv"
):
    device = "cpu"
    numerical_gradients = [grad for grad in numerical_gradients]
    backward_gradients = [grad.to(device) for grad in backward_gradients]

    backward_grad_flat = flatten_parameters(backward_gradients).cpu().numpy()

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter Index", "Numerical Gradient", "Backward Gradient", "Relative Error"])

        for i in range(len(param_indices)):
            num_grad = numerical_gradients[i]
            idx = param_indices[i]
            back_grad = backward_grad_flat[idx]
            relative_error = np.linalg.norm(back_grad - num_grad) / (
                np.linalg.norm(back_grad) + np.linalg.norm(num_grad) + 1e-8
            )
            writer.writerow([idx, num_grad, back_grad, relative_error])

    print(f"Gradient comparison saved to {csv_filename}")


if __name__ == "__main__":
    metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
    batch_size = 1
    N = 1
    patch_size = 224
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module = NDPI_DataModule(metadata_file, batch_size, num_workers=64)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    model = SpecimenClassifier(N, patch_size=patch_size, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    batch = next(iter(train_loader))
    topview_image, search_view_indexible, class_index = batch
    input_data = (topview_image.to(device), search_view_indexible)
    class_index = class_index.to(device)

    print("Computing numerical gradient...")

    model.to("cpu")
    input_data = (input_data[0].to("cpu"), input_data[1])
    class_index = class_index.to("cpu")

    N_params = 100  # Number of randomly selected parameters for numerical gradient calculation
    numerical_gradients, param_indices = compute_numerical_gradient(
        model, input_data, class_index, loss_fn, n_params=N_params
    )

    print("Computing backward gradient...")
    backward_gradients = compute_backward_gradient(model, input_data, class_index, loss_fn)

    print("Comparing gradients...")
    compare_gradients(numerical_gradients, backward_gradients, param_indices)
