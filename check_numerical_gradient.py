import torch
import numpy as np
from scipy.optimize import approx_fprime


def loss_fn_wrapper(model, params_flat, input_data, target_data, loss_fn):
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


def compute_numerical_gradient(model, input_data, target_data, loss_fn, epsilon=1e-5):
    # Flatten model parameters into a 1D numpy array
    params_flat = np.concatenate(
        [param.detach().numpy().flatten() for param in model.parameters()]
    )

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
    output = model(input_data)
    loss = loss_fn(output, target_data)

    # Zero the gradients before the backward pass
    model.zero_grad()

    # Perform backward pass to compute gradients
    loss.backward()

    # Store the gradients
    backward_gradients = [param.grad.clone() for param in model.parameters()]

    return backward_gradients


def compare_gradients(numerical_gradients, backward_gradients):
    # Compare numerical and backward gradients
    for idx, (num_grad, back_grad) in enumerate(
        zip(numerical_gradients, backward_gradients)
    ):
        relative_error = torch.norm(back_grad - num_grad) / (
            torch.norm(back_grad) + torch.norm(num_grad)
        )
        print(f"Parameter {idx}: Relative error: {relative_error}")


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
    data_module = NDPI_DataModule(metadata_file, batch_size, num_workers=64)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Instantiate model, loss, optimizer, and metrics
    model = SpecimenClassifier(N, patch_size=patch_size, num_classes=num_classes).to(
        device
    )
    loss_fn = nn.CrossEntropyLoss()

    # get the first batch of data
    batch = next(iter(train_loader))
    topview_image, search_view_indexible, class_index = batch

    print("Computing numerical gradient...")

    # Compute numerical gradients
    numerical_gradients = compute_numerical_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    print("Computing backward gradient...")

    # Compute backward gradients
    backward_gradients = compute_backward_gradient(
        model, (topview_image, search_view_indexible), class_index, loss_fn
    )

    print("Comparing gradients...")

    # Compare gradients
    compare_gradients(numerical_gradients, backward_gradients)
