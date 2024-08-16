import torch


class DifferentiableIndex2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexable_obj, indices):
        device = indices.device  # Ensure the device where indices are located

        # Floor and ceil for each dimension
        indices_y_floor = torch.floor(indices[:, 0]).long()
        indices_x_floor = torch.floor(indices[:, 1]).long()
        indices_y_ceil = torch.ceil(indices[:, 0]).long()
        indices_x_ceil = torch.ceil(indices[:, 1]).long()

        # Extract values at floor and ceil indices
        values_floor_floor = torch.tensor(
            [
                indexable_obj[indices_y_floor[i], indices_x_floor[i]]
                for i in range(len(indices))
            ],
            dtype=torch.float32,
        ).to(device)
        values_floor_ceil = torch.tensor(
            [
                indexable_obj[indices_y_floor[i], indices_x_ceil[i]]
                for i in range(len(indices))
            ],
            dtype=torch.float32,
        ).to(device)
        values_ceil_floor = torch.tensor(
            [
                indexable_obj[indices_y_ceil[i], indices_x_floor[i]]
                for i in range(len(indices))
            ],
            dtype=torch.float32,
        ).to(device)
        values_ceil_ceil = torch.tensor(
            [
                indexable_obj[indices_y_ceil[i], indices_x_ceil[i]]
                for i in range(len(indices))
            ],
            dtype=torch.float32,
        ).to(device)

        # Save tensors for the backward pass
        ctx.save_for_backward(
            indices,
            values_floor_floor,
            values_floor_ceil,
            values_ceil_floor,
            values_ceil_ceil,
        )

        # Bilinear interpolation
        weights_y_floor = indices[:, 0] - indices_y_floor.float().to(device)
        weights_y_ceil = indices_y_ceil.float().to(device) - indices[:, 0]
        weights_x_floor = indices[:, 1] - indices_x_floor.float().to(device)
        weights_x_ceil = indices_x_ceil.float().to(device) - indices[:, 1]

        interpolated_y_floor = (
            weights_x_ceil * values_floor_floor + weights_x_floor * values_floor_ceil
        )
        interpolated_y_ceil = (
            weights_x_ceil * values_ceil_floor + weights_x_floor * values_ceil_ceil
        )

        output = (
            weights_y_ceil * interpolated_y_floor
            + weights_y_floor * interpolated_y_ceil
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            indices,
            values_floor_floor,
            values_floor_ceil,
            values_ceil_floor,
            values_ceil_ceil,
        ) = ctx.saved_tensors

        # Calculate gradients for indices
        grad_indices_y = (
            (values_ceil_floor + values_ceil_ceil)
            - (values_floor_floor + values_floor_ceil)
        ) * grad_output
        grad_indices_x = (
            (values_floor_ceil + values_ceil_ceil)
            - (values_floor_floor + values_ceil_floor)
        ) * grad_output

        # Combine gradients for y and x
        grad_indices = torch.stack([grad_indices_y, grad_indices_x], dim=1)

        # No gradient for indexable_obj
        grad_indexable_obj = None

        return grad_indexable_obj, grad_indices


class DifferentiableIndex2DBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indexable_objs, indices_batch):
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

            # Floor and ceil indices
            indices_y_floor = torch.floor(indices[:, 0]).long()
            indices_x_floor = torch.floor(indices[:, 1]).long()
            indices_y_ceil = torch.ceil(indices[:, 0]).long()
            indices_x_ceil = torch.ceil(indices[:, 1]).long()

            # Extract values at floor and ceil indices
            values_floor_floor = torch.tensor(
                [
                    indexable_obj[indices_y_floor[j], indices_x_floor[j]]
                    for j in range(len(indices))
                ],
                dtype=torch.float32,
            ).to(device)
            values_floor_ceil = torch.tensor(
                [
                    indexable_obj[indices_y_floor[j], indices_x_ceil[j]]
                    for j in range(len(indices))
                ],
                dtype=torch.float32,
            ).to(device)
            values_ceil_floor = torch.tensor(
                [
                    indexable_obj[indices_y_ceil[j], indices_x_floor[j]]
                    for j in range(len(indices))
                ],
                dtype=torch.float32,
            ).to(device)
            values_ceil_ceil = torch.tensor(
                [
                    indexable_obj[indices_y_ceil[j], indices_x_ceil[j]]
                    for j in range(len(indices))
                ],
                dtype=torch.float32,
            ).to(device)

            # Save tensors for backward pass
            saved_tensors.extend(
                [
                    indices,
                    values_floor_floor,
                    values_floor_ceil,
                    values_ceil_floor,
                    values_ceil_ceil,
                ]
            )

            # Bilinear interpolation
            weights_y_floor = indices[:, 0] - indices_y_floor.float().to(device)
            weights_y_ceil = indices_y_ceil.float().to(device) - indices[:, 0]
            weights_x_floor = indices[:, 1] - indices_x_floor.float().to(device)
            weights_x_ceil = indices_x_ceil.float().to(device) - indices[:, 1]

            interpolated_y_floor = (
                weights_x_ceil * values_floor_floor
                + weights_x_floor * values_floor_ceil
            )
            interpolated_y_ceil = (
                weights_x_ceil * values_ceil_floor + weights_x_floor * values_ceil_ceil
            )

            output = (
                weights_y_ceil * interpolated_y_floor
                + weights_y_floor * interpolated_y_ceil
            )
            output_batch.append(output)

        # Save all necessary tensors for the backward pass
        ctx.save_for_backward(*saved_tensors)

        # Stack output for the batch
        return torch.stack(output_batch)

    @staticmethod
    def backward(ctx, grad_output_batch):
        saved_tensors = ctx.saved_tensors

        # Gradient container for indices_batch
        grad_indices_batch = []

        # Each item in the batch has 5 saved tensors
        num_saved_tensors_per_item = 5

        # Process each item in the batch
        for i in range(0, len(saved_tensors), num_saved_tensors_per_item):
            (
                indices,
                values_floor_floor,
                values_floor_ceil,
                values_ceil_floor,
                values_ceil_ceil,
            ) = saved_tensors[i : i + num_saved_tensors_per_item]
            grad_output = grad_output_batch[i // num_saved_tensors_per_item]

            # Calculate gradients for indices
            grad_indices_y = (
                (values_ceil_floor + values_ceil_ceil)
                - (values_floor_floor + values_floor_ceil)
            ) * grad_output
            grad_indices_x = (
                (values_floor_ceil + values_ceil_ceil)
                - (values_floor_floor + values_ceil_floor)
            ) * grad_output

            # Combine gradients for y and x
            grad_indices = torch.stack([grad_indices_y, grad_indices_x], dim=1)
            grad_indices_batch.append(grad_indices)

        # No gradient for indexable_objs
        grad_indexable_objs = None

        # Stack gradients for the batch
        return grad_indexable_objs, torch.stack(grad_indices_batch)


# Example use in your model
def differentiable_index_2d_batch(indexable_objs, indices_batch):
    return DifferentiableIndex2DBatchFunction.apply(indexable_objs, indices_batch)