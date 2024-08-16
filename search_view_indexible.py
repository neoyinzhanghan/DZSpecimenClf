import openslide
import numpy as np
import torch


class SearchViewIndexible:
    """A class to represent a searchable view of a WSI. It is not tensor representation of the search view or the WSI.
    It is supposed to just an indexible representation of the search view that only be sparsely sampled on the CPU during training and inference.

    === Class Attributes ===
    --wsi_path: str
    --search_view_level: int
    --search_to_top_downsample_factor: int
    --search_view_height: int
    --search_view_width: int
    """

    def __init__(
        self, wsi_path, search_view_level=3, search_to_top_downsample_factor=16
    ) -> None:
        self.wsi_path = wsi_path

        self.search_view_level = search_view_level
        self.search_to_top_downsample_factor = search_to_top_downsample_factor

        self.search_view_height, self.search_view_width = openslide.OpenSlide(
            self.wsi_path
        ).level_dimensions[self.search_view_level]

    def __getitem__(self, idx):
        """Retrieve a single pixel from the slide based on the (y, x) coordinates.

        Args:
        idx (tuple): A tuple of (y, x) defining the pixel coordinates.

        Returns:
        np.array: The RGB values of the extracted pixel as a numpy array.
        """

        # assert that y is in the range of the search view height and x is in the range of the search view width
        assert (
            0 <= idx[0] < self.search_view_height
        ), f"y: {idx[0]} is out of range of the search view height: {self.search_view_height}"
        assert (
            0 <= idx[1] < self.search_view_width
        ), f"x: {idx[1]} is out of range of the search view width: {self.search_view_width}"

        y, x = idx
        # Extracting a region of 1x1 pixels
        region = self.slide.read_region(
            (
                int(x * (2**self.search_view_level)),
                int(y * (2**self.search_view_level)),
            ),
            self.search_view_level,
            (1, 1),
        )
        # Convert to numpy array, remove alpha channel, and reshape
        pixel_values = np.array(region)[:, :, :3]  # shape will be (1, 1, 3)
        pixel_values = pixel_values.reshape(3)  # reshape to (3,)
        # Convert to a torch tensor
        return torch.tensor(pixel_values, dtype=torch.uint8)
