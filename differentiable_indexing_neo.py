import torch


class DifferentiableCrop2DBatchFunction(torch.autograd.Function):

    # @staticmethod
    # def forward(ctx, indexable_objs, indices_batch, patch_size=224):
    #     device = indices_batch.device # Ensure device consistency

    #     output_batch = []
    #     saved_tensors = []

    #     for i in range(len(indexable_objs)):

    #         indices = indices_batch[i]
    #         indexable_obj = indexable_objs[i]

    #         central_indices_y_floor = torch.floor(indices[:, 0]).long()
    #         central_indices_x_floor = torch.floor(indices[:, 1]).long()
    #         central_indices_y_ceil = torch.ceil(indices[:, 0]).long()
    #         central_indices_x_ceil = torch.ceil(indices[:, 1]).long()

    #         #TODO: Why are we subtracting by half of the patch size? TL ==  topleft
    #         # calculate the TL_indices floor and ceil by subtracting half of the patch size
    #         TL_indices_y_floor = central_indices_y_floor - patch_size // 2
    #         TL_indices_x_floor = central_indices_x_floor - patch_size // 2
    #         TL_indices_y_ceil = central_indices_y_ceil - patch_size // 2
    #         TL_indices_x_ceil = central_indices_x_ceil - patch_size // 2

    @staticmethod
    def forward(ctx, indexable_objs, indices_batch, patch_size=224):
        device = indices_batch.device  # Ensure device consistency

        # Ensure that indexable_objs has the same length as the batch dimension
        assert (
            len(indexable_objs) == indices_batch.shape[0]
        ), f"indexable_objs length {len(indexable_objs)} must match batch dimension {indices_batch.shape[0]}"

        # Result container
        output_batch = []
        saved_tensors = []

        # Process each item in the batch
        for i in range(len(indexable_objs)):
            indices = indices_batch[i]
            indexable_obj = indexable_objs[i]

            central_indices_y_floor = torch.floor(indices[:, 0]).long()
            central_indices_x_floor = torch.floor(indices[:, 1]).long()
            central_indices_y_ceil = torch.ceil(indices[:, 0]).long()
            central_indices_x_ceil = torch.ceil(indices[:, 1]).long()

            # TODO: Why are we subtracting by half of the patch size? TL ==  topleft
            # calculate the TL_indices floor and ceil by subtracting half of the patch size
            TL_indices_y_floor = central_indices_y_floor - patch_size // 2
            TL_indices_x_floor = central_indices_x_floor - patch_size // 2
            TL_indices_y_ceil = central_indices_y_ceil - patch_size // 2
            TL_indices_x_ceil = central_indices_x_ceil - patch_size // 2

            # extract the patch_size x patch_size patches
            # use indexable_obj.crop(self, TL_x, TL_y, patch_size=224) method, which returns a tensor of shape [patch_size, patch_size, 3]
            # the shape of patches_floor_floor is (len(indices), patch_size, patch_size, 3)
            patches_floor_floor = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_floor[j],
                        TL_indices_y_floor[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            assert patches_floor_floor.shape == (
                len(indices),
                patch_size,
                patch_size,
                3,
            )

            patches_ceil_floor = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_ceil[j],
                        TL_indices_y_floor[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            patches_floor_ceil = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_floor[j],
                        TL_indices_y_ceil[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            patches_ceil_ceil = torch.stack(
                [
                    indexable_obj.crop(
                        TL_indices_x_ceil[j],
                        TL_indices_y_ceil[j],
                        patch_size=patch_size,
                    )
                    for j in range(len(indices))
                ],
                dim=0,
            )

            # Bilinear interpolation
            weights_y_ceil = indices[:, 0] - TL_indices_y_floor.float().to(device)
            weights_y_floor = TL_indices_y_ceil.float().to(device) - indices[:, 0]
            weights_x_ceil = indices[:, 1] - TL_indices_x_floor.float().to(device)
            weights_x_floor = TL_indices_x_ceil.float().to(device) - indices[:, 1]

            weights_x_ceil = weights_x_ceil.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]
            weights_x_floor = weights_x_floor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]
            weights_y_ceil = weights_y_ceil.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]
            weights_y_floor = weights_y_floor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # Shape: [len(indices), 1, 1, 1]

            # move patches_floor_floor to the same device as weights_x_ceil
            patches_floor_floor = patches_floor_floor.to(weights_x_ceil.device)
            # do the same for the rest of the patches
            patches_floor_ceil = patches_floor_ceil.to(weights_x_ceil.device)
            patches_ceil_floor = patches_ceil_floor.to(weights_x_ceil.device)
            patches_ceil_ceil = patches_ceil_ceil.to(weights_x_ceil.device)

            # print(f"Shape of patches_floor_floor: {patches_floor_floor.shape}")
            # print(f"Shape of weights_x_ceil: {weights_x_ceil.shape}")

            output = (
                weights_x_floor * weights_y_floor * patches_floor_floor
                + weights_x_ceil * weights_y_floor * patches_ceil_floor
                + weights_x_floor * weights_y_ceil * patches_floor_ceil
                + weights_x_ceil * weights_y_ceil * patches_ceil_ceil
            )

            output_batch.append(output)

            # Save tensors for backward pass
            saved_tensors.extend(
                [
                    indices,
                    patches_floor_floor,
                    patches_floor_ceil,
                    patches_ceil_floor,
                    patches_ceil_ceil,
                    weights_x_floor,
                    weights_x_ceil,
                    weights_y_floor,
                    weights_y_ceil,
                ]
            )

        # Save all necessary tensors for the backward pass
        ctx.save_for_backward(*saved_tensors)

        output = torch.stack(output_batch, dim=0)
        assert output.shape == (
            len(indexable_objs),
            len(indices_batch[0]),
            patch_size,
            patch_size,
            3,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved_tensors = ctx.saved_tensors

        # Gradient container for indices_batch
        grad_indices_batch = []

        # Each item in the batch has 5 saved tensors
        num_saved_tensors_per_item = 9

        batch_size = grad_output.shape[0]

        for i in range(batch_size):
            start_idx = i * num_saved_tensors_per_item
            end_idx = (
                start_idx + num_saved_tensors_per_item
            )  # this is just to calculate the actual position of the saved_tensors in the big list

            (
                indices,
                patches_floor_floor,
                patches_floor_ceil,
                patches_ceil_floor,
                patches_ceil_ceil,
                weights_x_floor,
                weights_x_ceil,
                weights_y_floor,
                weights_y_ceil,
            ) = saved_tensors[start_idx:end_idx]

            grad_output_item = grad_output[i]

            # Ensure that tensors are on the correct device
            patches_floor_floor = patches_floor_floor.to(grad_output_item.device)
            patches_floor_ceil = patches_floor_ceil.to(grad_output_item.device)
            patches_ceil_floor = patches_ceil_floor.to(grad_output_item.device)
            patches_ceil_ceil = patches_ceil_ceil.to(grad_output_item.device)

            # # Calculate gradients for indices
            # grad_indices_y_mat = (patches_ceil_floor + patches_ceil_ceil) - (
            #     patches_floor_floor + patches_floor_ceil
            # )
            # grad_indices_x_mat = (patches_floor_ceil + patches_ceil_ceil) - (
            #     patches_floor_floor + patches_ceil_floor
            # )

            # grad_indices_mat = y_inter_ceil - y_inter_floor
            grad_indices_y_mat = weights_x_ceil * (
                patches_ceil_ceil - patches_ceil_floor
            ) + weights_x_floor * (patches_floor_ceil - patches_floor_floor)

            # grad_indices_mat = x_inter_ceil - x_inter_floor
            grad_indices_x_mat = weights_y_ceil * (
                patches_ceil_ceil - patches_floor_ceil
            ) + weights_y_floor * (patches_ceil_floor - patches_floor_floor)

            # print(f"Shape of grad_indices_y_mat: {grad_indices_y_mat.shape}")

            # stack the gradients for y and x along the dim 1
            grad_indices_mat = torch.stack(
                [grad_indices_y_mat, grad_indices_x_mat], dim=1
            )
            # print(f"Shape of grad_indices_mat: {grad_indices_mat.shape}")

            # Ensure indices_mat is a float tensor
            if grad_indices_mat.dtype != torch.float32:
                grad_indices_mat = grad_indices_mat.float()

            # Ensure grad_output is a float tensor
            if grad_output_item.dtype != torch.float32:
                grad_output_item = grad_output_item.float()

            # print("grad_indices_mat shape: ", grad_indices_mat.shape)
            # print("grad_output_item shape: ", grad_output_item.shape)

            # Flatten the spatial dimensions (224 * 224 * 3 = 150528)
            grad_indices_mat_flat = grad_indices_mat.view(
                8, 2, -1
            )  # Shape: [8, 2, 150528] 150528 or would be whatever flatten dimension of the patch image is
            grad_output_item_flat = grad_output_item.view(8, -1)  # Shape: [8, 150528]

            # Perform batch matrix multiplication
            output = torch.bmm(
                grad_indices_mat_flat, grad_output_item_flat.unsqueeze(2)
            )  # Shape: [8, 2, 1]

            # Remove the last dimension to get the desired shape [8, 2]
            output = output.squeeze(2)

            grad_indices_batch.append(output)

        grad_indices_stacked = torch.stack(grad_indices_batch, dim=0)

        # No gradient for indexable_objs
        grad_indexable_objs = None

        # Stack gradients for the batch
        return (
            grad_indexable_objs,
            grad_indices_stacked,
            None,
        )


def differentiable_crop_2d_batch(indexable_objs, indices_batch, patch_size=224):
    return DifferentiableCrop2DBatchFunction.apply(
        indexable_objs, indices_batch, patch_size
    )
