import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights
from differentiable_indexing import differentiable_index_2d_batch


class DZSpecimenClf(nn.Module):
    def __init__(self, N, k, num_classes=2):
        super(DZSpecimenClf, self).__init__()
        # Load the pretrained ResNeXt50 model
        self.resnext50 = models.resnext50_32x4d(pretrained=False)

        self.resnext50.fc = nn.Linear(self.resnext50.fc.in_features, N * k * 2)

        # Define a sigmoid activation layer
        self.sigmoid = nn.Sigmoid()

        # initialize a trainable tensor of shape (N, k, 1)
        self.N = N
        self.k = k

        # initailize the trainable tensor
        self.weights = nn.Parameter(torch.rand((self.N, self.k, 1), requires_grad=True))

        # initialize a trainable three numbers each corresponding to each of the RGB channels
        self.rgb_weights = nn.Parameter(torch.rand((3), requires_grad=True))

        # now have a last forward from N to num_classes
        self.fc = nn.Linear(N, num_classes)

    def forward(self, topview_image_tensor, search_view_indexibles):
        # Pass input through the feature extractor part
        x = self.resnext50(topview_image_tensor)

        # assert that the output is of the correct shape
        assert (
            x.shape[1]
            == search_view_indexibles.search_view_height
            * search_view_indexibles.search_view_width
            * 2
        ), f"Output shape is {x.shape}, rather than the expected ({search_view_indexibles.search_view_height * search_view_indexibles.search_view_width * 2})"

        x = x.view(x.size(0), -1, 2)

        # assert that the output is of the correct shape
        assert (
            x.shape[1] == self.N * self.k and x.shape[2] == 2
        ), f"Output shape is {x.shape}, rather than the expected ({self.N * self.k}, 2)"

        # apply the sigmoid activation
        x = self.sigmoid(x)

        search_view_heights = [
            search_view_indexible.search_view_height
            for search_view_indexible in search_view_indexibles
        ]
        search_view_widths = [
            search_view_indexible.search_view_width
            for search_view_indexible in search_view_indexibles
        ]

        assert (
            len(search_view_heights)
            == len(search_view_widths)
            == len(search_view_indexibles)
            == x.shape[0]
        ), f"Batch dim / length of search_view_heights: {len(search_view_heights)}, search_view_widths: {len(search_view_widths)}, search_view_indexibles: {len(search_view_indexibles)}, x: {x.shape[0]}"

        # x is a bunch of y, x coordinates there are b, N*k of them, multiply y by the search view height and x by the search view width
        # Scale x by multiplying the y and x coordinates by the respective dimensions
        # First column of x are y coordinates, second column are x coordinates

        # multiple x[b: :0] parallelly across the batch dimension by the search view height
        x[:, :, 0] = x[:, :, 0] * torch.tensor(search_view_heights).view(-1, 1, 1)
        # multiple x[b: :1] parallelly across the batch dimension by the search view width
        x[:, :, 1] = x[:, :, 1] * torch.tensor(search_view_widths).view(-1, 1, 1)

        # apply differentiable indexing
        x = differentiable_index_2d_batch(search_view_indexibles, x)

        # assert the indexing_output is of the correct shape
        assert (
            x.shape[1] == self.N * self.k
        ), f"Output shape is {x.shape}, rather than the expected ({self.N * self.k})"
        assert (
            x.shape[2] == 3
        ), f"Output shape is {x.shape}, rather than the expected (3)"

        # reshape the indexing_output to have the shape (N, k, 3)
        x = x.view(-1, self.N, self.k, 3)

        # assert that the weights are of the correct shape
        assert self.weights.shape == (
            self.N,
            self.k,
            1,
        ), f"Output shape is {self.weights.shape}, rather than the expected ({self.N}, {self.k}, 1)"

        # multiply the indexing_output by the weights
        x = x * self.weights

        # sum the output along the k, and the output should be of shape (b, N, 3)
        x = torch.sum(x, dim=2)

        assert x.shape == (
            x.shape[0],
            self.N,
            3,
        ), f"Output shape is {x.shape}, rather than the expected ({x.shape[0]}, {self.N}, 3)"

        # use the rgb_weights to multiply the output across the 3 channels and then sum across the channels to have output of shape (b, N)
        x = x * self.rgb_weights
        x = torch.sum(x, dim=2)

        # now make sure the output has shape (b,N)
        assert x.shape == (
            x.shape[0],
            self.N,
        ), f"Output shape is {x.shape}, rather than the expected ({x.shape[0]}, {self.N})"

        # pass the output through the final fully connected layer
        x = self.fc(x)

        return x
