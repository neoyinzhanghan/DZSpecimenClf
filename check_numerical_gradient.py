import torch
import torch.nn.functional as F
import numpy as np
from DZSpecimenClf import DZSpecimenClf

def compute_numerical_gradient(model, topview_image_tensor, search_view_indexibles, param_name, param, epsilon=1e-5):
    """
    Compute the numerical gradient of a specific parameter in the model.

    Args:
        model (nn.Module): The model containing the parameter.
        topview_image_tensor (torch.Tensor): Input tensor for the model.
        search_view_indexibles (list): List of search view indexibles.
        param_name (str): Name of the parameter for gradient checking.
        param (torch.Tensor): Parameter tensor for gradient checking.
        epsilon (float): Small perturbation for numerical gradient computation.

    Returns:
        torch.Tensor: Numerical gradient of the parameter.
    """
    numerical_grad = torch.zeros_like(param)

    # Iterate over all elements in the parameter tensor
    for i in range(param.numel()):
        # Save the original value
        original_value = param.view(-1)[i].item()

        # Perturb parameter positively
        param.view(-1)[i] = original_value + epsilon
        output_plus = model(topview_image_tensor, search_view_indexibles)
        output_plus = F.log_softmax(output_plus, dim=1)
        loss_plus = F.nll_loss(output_plus, torch.zeros_like(output_plus[:, 0], dtype=torch.long))

        # Perturb parameter negatively
        param.view(-1)[i] = original_value - epsilon
        output_minus = model(topview_image_tensor, search_view_indexibles)
        output_minus = F.log_softmax(output_minus, dim=1)
        loss_minus = F.nll_loss(output_minus, torch.zeros_like(output_minus[:, 0], dtype=torch.long))

        # Restore original value
        param.view(-1)[i] = original_value

        # Compute numerical gradient: (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
        numerical_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return numerical_grad

def compare_gradients(model, topview_image_tensor, search_view_indexibles, epsilon=1e-5):
    """
    Compare the backward and numerical gradients for each custom parameter in the model.

    Args:
        model (nn.Module): The model to check gradients.
        topview_image_tensor (torch.Tensor): Input tensor for the model.
        search_view_indexibles (list): List of search view indexibles.
        epsilon (float): Small perturbation for numerical gradient computation.
    """
    # Zero all gradients
    model.zero_grad()

    # Perform a forward pass and compute backward gradients
    output = model(topview_image_tensor, search_view_indexibles)
    output = F.log_softmax(output, dim=1)  # Apply log-softmax
    loss = F.nll_loss(output, torch.zeros_like(output[:, 0], dtype=torch.long))
    loss.backward()

    # Check gradients for custom parameters only
    for name, param in model.named_parameters():
        if "resnext50" not in name and param.grad is not None:
            backward_grad = param.grad
            numerical_grad = compute_numerical_gradient(model, topview_image_tensor, search_view_indexibles, name, param, epsilon)

            # Calculate the relative error
            relative_error = (backward_grad - numerical_grad).abs() / (numerical_grad.abs() + backward_grad.abs() + 1e-8)

            print(f"Parameter: {name}")
            print(f"Backward Gradient: {backward_grad}")
            print(f"Numerical Gradient: {numerical_grad}")
            print(f"Relative Error: {relative_error}\n")

# Example usage
if __name__ == "__main__":
    # Example setup
    N, k, num_classes = 2, 2, 2
    model = DZSpecimenClf(N, k, num_classes)

    # Create dummy inputs
    topview_image_tensor = torch.randn(1, 3, 224, 224)  # Example top-view image tensor
    search_view_indexibles = [None]  # Replace with actual objects or dummy placeholders

    # Compare gradients
    compare_gradients(model, topview_image_tensor, search_view_indexibles)
