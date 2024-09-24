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
        self.sigmoid = nn.Sigmoid()
        self.N = N
        self.patch_size = patch_size

        # Initialize weights using random Gaussian initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Gaussian (normal) initialization with mean=0 and std=0.01
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)  # Bias initialized to zero

    def forward(self, topview_image_tensor, search_view_indexibles):
        # Downsample the topview_image_tensor to a 4x4 image
        topview_image_tensor_downsampled = F.interpolate(
            topview_image_tensor, size=(4, 4), mode="bilinear"
        )

        # Flatten the downsampled tensor
        topview_image_tensor_flattened = topview_image_tensor_downsampled.view(
            topview_image_tensor_downsampled.size(0), -1
        )

        # Pass through the first linear layer
        x = self.topview_linear(topview_image_tensor_flattened)

        # Reshape and apply sigmoid activation
        x = x.view(x.size(0), -1, 2)
        assert (
            x.shape[1] == self.N and x.shape[2] == 2
        ), f"Expected shape ({self.N}, 2), but got {x.shape}"
        x = self.sigmoid(x)

        # Scale coordinates according to search view dimensions
        search_view_heights = [
            svi.search_view_height - 1 for svi in search_view_indexibles
        ]
        search_view_widths = [
            svi.search_view_width - 1 for svi in search_view_indexibles
        ]
        padded_search_view_heights = [
            svh - self.patch_size for svh in search_view_heights
        ]
        padded_search_view_widths = [
            svw - self.patch_size for svw in search_view_widths
        ]

        assert (
            len(search_view_heights)
            == len(search_view_widths)
            == len(search_view_indexibles)
            == x.shape[0]
        )

        search_view_heights_tensor = (
            torch.tensor(padded_search_view_heights).view(-1, 1, 1).to(x.device)
        )
        search_view_widths_tensor = (
            torch.tensor(padded_search_view_widths).view(-1, 1, 1).to(x.device)
        )

        # Scale the x and y coordinates
        x_scaled = (x[..., 0].unsqueeze(-1) * search_view_heights_tensor).squeeze(-1)
        y_scaled = (x[..., 1].unsqueeze(-1) * search_view_widths_tensor).squeeze(-1)

        # Offset by patch size
        x_scaled += self.patch_size // 2
        y_scaled += self.patch_size // 2

        # Stack the coordinates and perform differentiable cropping
        xy = torch.stack([x_scaled, y_scaled], dim=-1)
        x = differentiable_crop_2d_batch(
            search_view_indexibles, xy, patch_size=self.patch_size
        )

        # Reshape for the final layer
        assert (
            x.shape[1] == self.N
            and x.shape[2] == self.patch_size
            and x.shape[3] == self.patch_size
            and x.shape[4] == 3
        )
        x = x.view(x.size(0), -1)

        # Pass through the final linear layer
        x = self.last_layer(x)

        return x
