import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights
from differentiable_indexing import differentiable_crop_2d_batch
from PIL import ImageDraw
from torchvision import transforms


class DZSpecimenClfToy(nn.Module):
    def __init__(self, N, patch_size=4, num_classes=2):
        super(DZSpecimenClfToy, self).__init__()
        self.topview_linear = nn.Linear(16 * 3, N * 2)

        self.last_layer = nn.Linear(N * 16 * 3, num_classes)
        self.num_classes = num_classes

        # Define a sigmoid activation layer
        self.sigmoid = nn.Sigmoid()

        # initialize a trainable tensor of shape (N, k, 1)
        self.N = N
        self.patch_size = patch_size

    def forward(self, topview_image_tensor, search_view_indexibles):
        # downsample the topview_image_tensor to a 4x4 image, note that the image has 3 channels
        topview_image_tensor = F.interpolate(
            topview_image_tensor, size=(4, 4), mode="bilinear"
        )

        # flatten the topview_image_tensor into a 1D tensor
        topview_image_tensor_flattened = topview_image_tensor.view(
            topview_image_tensor.size(0), -1
        )

        x = self.topview_linear(topview_image_tensor_flattened)

        x = x.view(
            x.size(0), -1, 2
        )  # now after reshaping, x should have shape [b, N, 2]

        # assert that the output is of the correct shape
        assert (
            x.shape[1] == self.N and x.shape[2] == 2
        ), f"Output shape is {x.shape}, rather than the expected ({self.N}, 2)"

        # apply the sigmoid activation
        x = self.sigmoid(x)

        search_view_heights = [
            search_view_indexible.search_view_height - 1
            for search_view_indexible in search_view_indexibles
        ]
        search_view_widths = [
            search_view_indexible.search_view_width - 1
            for search_view_indexible in search_view_indexibles
        ]

        # padded_search_view_heights will be search_view_heights subtracted by patch_size
        padded_search_view_heights = [
            search_view_height - self.patch_size
            for search_view_height in search_view_heights
        ]
        # padded_search_view_widths will be search_view_widths subtracted by patch_size
        padded_search_view_widths = [
            search_view_width - self.patch_size
            for search_view_width in search_view_widths
        ]

        assert (
            len(search_view_heights)
            == len(search_view_widths)
            == len(search_view_indexibles)
            == x.shape[0]
        ), f"Batch dim / length of search_view_heights: {len(search_view_heights)}, search_view_widths: {len(search_view_widths)}, search_view_indexibles: {len(search_view_indexibles)}, x: {x.shape[0]}"

        search_view_heights_tensor = (
            torch.tensor(padded_search_view_heights).view(-1, 1, 1).to(x.device)
        )
        search_view_widths_tensor = (
            torch.tensor(padded_search_view_widths).view(-1, 1, 1).to(x.device)
        )
        # x is a bunch of y, x coordinates there are b, N*k of them, multiply y by the search view height and x by the search view width
        # Scale x by multiplying the y and x coordinates by the respective dimensions
        # First column of x are y coordinates, second column are x coordinates

        x_scaled = (x[..., 0].unsqueeze(-1) * search_view_heights_tensor).squeeze(-1)
        y_scaled = (x[..., 1].unsqueeze(-1) * search_view_widths_tensor).squeeze(-1)

        # now add patch_size // 2 to the x_scaled and y_scaled tensors
        x_scaled = x_scaled + self.patch_size // 2
        y_scaled = y_scaled + self.patch_size // 2

        # now stack the x_scaled and y_scaled tensors along the last dimension
        xy = torch.stack([x_scaled, y_scaled], dim=-1)

        # Continue with x_scaled instead of x
        x = differentiable_crop_2d_batch(search_view_indexibles, xy)

        # assert that x has shape [b, N, patch_size, patch_size, 3]
        assert (
            x.shape[1] == self.N
            and x.shape[2] == self.patch_size
            and x.shape[3] == self.patch_size
            and x.shape[4] == 3
        ), f"Output shape is {x.shape}, rather than the expected (b, N, {self.patch_size}, {self.patch_size}, 3)"

        # now reshape x to have shape [b, N*3 * patch_size * patch_size]
        x = x.view(x.size(0), -1)

        x = self.last_layer(x)

        return x
