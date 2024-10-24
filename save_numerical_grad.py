import torch
import numpy as np
import json
from scipy.optimize import approx_fprime

def compute_numerical_gradient(model, input_data, target_data, loss_fn, epsilon=1e-5):
    # Flatten model parameters into a 1D numpy array
    params_flat = np.concatenate(
        [param.detach().cpu().numpy().flatten() for param in model.parameters()]
    )

    # Define the loss function wrapper
    def loss_fn_wrapper(params_flat):
        """
        Given a model, a flattened version of its parameters, input data, target data, and a loss function, this function
        will set the parameters of the model to the unflattened version of params_flat, compute the loss using the input
        data and target data, and return the loss.
        """
        # Unflatten parameters and set them to the model
        start_idx = 0
        for param in model.parameters():
            param_shape = param.shape
            numel = param.numel()
            param.data = torch.tensor(
                params_flat[start_idx : start_idx + numel].reshape(param_shape),
                dtype=param.dtype,
            )
            start_idx += numel

        # Compute the loss
        topview_image, search_view_indexible = input_data
        output = model(topview_image, search_view_indexible)
        loss = loss_fn(output, target_data).item()
        return loss

    # Compute numerical gradient using approx_fprime
    numerical_gradient_flat = approx_fprime(params_flat, loss_fn_wrapper, epsilon)

    # Reshape numerical gradient back to model parameter shapes
    numerical_gradients = []
    start_idx = 0
    for param in model.parameters():
        numel = param.numel()
        grad_shape = param.shape
        numerical_gradients.append(
            torch.tensor(
                numerical_gradient_flat[start_idx : start_idx + numel].reshape(
                    grad_shape
                )
            )
        )
        start_idx += numel

    return numerical_gradients


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


def save_gradients_to_json(gradients, file_name):
    gradients_dict = {}
    for idx, grad in enumerate(gradients):
        gradients_dict[f"Parameter_{idx}"] = grad.flatten().cpu().numpy().tolist()

    with open(file_name, mode="w") as file:
        json.dump(gradients_dict, file, indent=4)


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

    # Get the first batch of data
    batch = next(iter(train_loader))
    topview_image, search_view_indexible, class_index = batch
    topview_image = topview_image.to(device)
    class_index = class_index.to(device)

    print("Computing numerical gradient...")

    # Move model and data to CPU for numerical gradient computation
    model.to("cpu")
    topview_image = topview_image.to("cpu")
    class_index = class_index.to("cpu")

    # Compute numerical gradients
    numerical_gradients = compute_numerical_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    # Save numerical gradients to a JSON file
    save_gradients_to_json(numerical_gradients, "numerical_gradients.json")

    print("Computing backward gradient...")

    # Move model and data back to the original device for backward gradient computation
    model.to(device)
    topview_image = topview_image.to(device)
    class_index = class_index.to(device)

    # Compute backward gradients
    backward_gradients = compute_backward_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    # Save backward gradients to a JSON file
    save_gradients_to_json(backward_gradients, "backward_gradients.json")
