from typing import Sequence, Callable, Literal

import numpy as np
import xarray as xr

import torch
from torch.utils.data import random_split, Dataset, DataLoader, SubsetRandomSampler


class XarrayDataset(Dataset):
    def __init__(
        self,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        target_realization_avg: bool = True,
        transform_x: Callable = None,
        transform_y: Callable = None,
        transform_x_args: dict = None,
        transform_y_args: dict = None,
        realization_as_channel: bool = False,
        output_realizations: Literal["deterministic", "ensemble"] = "deterministic", # deterministic -> output R = 1, ensemble -> output R = input R
    ):
        """
        input_ds: xarray.Dataset
        target_ds: xarray.Dataset
        transform_x: callable with signature (x, **kwargs) -> x (for input)
        transform_y: callable with signature (y, **kwargs) -> y (for target)
        transform_x_args: dict of keyword args to pass to transform_x
        transform_y_args: dict of keyword args to pass to transform_y
        realization_as_channel: if True, treat the realization dimension R as channels (C), so output shape is (C=R,T,H,W).
                                If False, treat realization as independent samples andmerge R with T and output shape is (C,T*R,H,W).
        output_realizations: if "deterministic", the target will be averaged across R (if it has R) to produce a deterministic target (C,T,H,W).
                             If "ensemble", the target will be repeated across R_in (the number of input realizations) and merged into channels.
                             Only relevant if realization_as_channel is True. If False, target R dimension is merged with T and this setting is ignored.
        """

        self.target_ds = target_ds # .load(scheduler="synchronous")
        self.input_ds = input_ds
        assert isinstance(self.input_ds, xr.Dataset) and isinstance(self.target_ds, xr.Dataset), f"Input ds type: {self.input_ds}, target ds type: {self.target_ds}"

        self.transform_x, self.transform_y = transform_x, transform_y
        self.transform_x_args = transform_x_args or {}
        self.transform_y_args = transform_y_args or {}

        # NumPy arrays with NaNs for missing data
        x_np = self._transpose_dims_ds_to_da(self.input_ds).to_numpy()
        assert len(x_np.shape) > 3 and len(x_np.shape) < 6, f"Input has shape {x_np.shape}"
        # if x_np.ndim == 5 and x_np.shape[1] == 1:
        #     x_np = x_np.squeeze(1)          # C,T,H,W squeeze R dimension if present
        mask_x_np = np.isfinite(x_np)       # True where valid, False where masked
        x_np_filled = np.where(mask_x_np, x_np, 0.0)

        y_np = self._transpose_dims_ds_to_da(self.target_ds).to_numpy()
        assert len(y_np.shape) > 3 and len(y_np.shape) < 6, f"Target has shape {y_np.shape}"
        # if y_np.ndim == 5 and y_np.shape[1] == 1:
        #     y_np = y_np.squeeze(1)
        mask_y_np = np.isfinite(y_np)
        y_np_filled = np.where(mask_y_np, y_np, 0.0)

        assert all(d > 0 for d in x_np.shape), f"Input has zero-sized dimension: {x_np.shape}"
        assert all(d > 0 for d in y_np.shape), f"Target has zero-sized dimension: {y_np.shape}"

        # print(f"Dataset x mean: {np.nanmean(x_np)}, std: {np.nanstd(x_np)}")
        # print(f"Dataset y mean: {np.nanmean(y_np)}, std: {np.nanstd(y_np)}")
        # print("input vars:", list(input_ds.data_vars))
        # print("target vars:", list(target_ds.data_vars))
        # print(f"Input shape: {x_np.shape}, target shape: {y_np.shape}")

        if realization_as_channel:
            print("Realization as channel branch")
            # X: merge R into C
            if x_np_filled.ndim == 5:  # (C,T,R,H,W)
                C, T, R, H, W = x_np_filled.shape
                x_np_filled_ = x_np_filled.reshape(C * R, T, H, W)          # (C*R,T,H,W)
                mask_x_np_   = mask_x_np.reshape(C * R, T, H, W)
            else:  # (C,T,H,W)
                x_np_filled_ = x_np_filled                                  # (C,T,H,W)
                mask_x_np_   = mask_x_np

            self.x      = torch.tensor(x_np_filled_, dtype=torch.float32).permute(1, 0, 2, 3)  # (T,Cin,H,W)
            self.x_mask = torch.tensor(mask_x_np_,   dtype=torch.bool).permute(1, 0, 2, 3)     # (T,Cin,H,W)

            if output_realizations=="deterministic":
                # Y: keep deterministic (always average if it has R, target_realization_avg ignored in this branch)
                if y_np_filled.ndim == 5:  # (C,T,R,H,W)
                    # If deterministic target is stored with R=1, this just squeezes/averages safely.
                    y_np_filled_ = np.nanmean(y_np_filled, axis=2)   # -> (C,T,H,W)
                    mask_y_np_   = np.any(mask_y_np, axis=2)         # -> (C,T,H,W)
                else:  # (C,T,H,W)
                    y_np_filled_ = y_np_filled
                    mask_y_np_   = mask_y_np

            elif output_realizations=="ensemble":
                # Y: ensemble target with R_in realizations,
                # then fold R into channels so y has Cout*R_in channels (matching model output convention).

                R_in = x_np_filled.shape[2] if x_np_filled.ndim == 5 else 1

                # Make deterministic base target y_det: (C, T, H, W)
                if y_np_filled.ndim == 5:  # (C,T,R,H,W) -> deterministic base
                    y_det = np.nanmean(y_np_filled, axis=2)   # (C,T,H,W)
                    m_det = np.any(mask_y_np, axis=2)         # (C,T,H,W)
                else:  # (C,T,H,W)
                    y_det = y_np_filled
                    m_det = mask_y_np

                # Expand across R_in: (C,T,H,W) -> (C,T,R_in,H,W)
                # Use repeat to physically materialize
                y_rep = np.repeat(y_det[:, :, None, :, :], R_in, axis=2)  # (C,T,R_in,H,W)
                m_rep = np.repeat(m_det[:, :, None, :, :], R_in, axis=2)  # (C,T,R_in,H,W)

                # Fold R into channels: (C,T,R,H,W) -> (C*R,T,H,W)
                C, T, R, H, W = y_rep.shape
                y_np_filled_ = y_rep.reshape(C * R, T, H, W)  # (Cout*R_in, T, H, W)
                mask_y_np_   = m_rep.reshape(C * R, T, H, W)  # (Cout*R_in, T, H, W)

            else:
                raise ValueError(f"Unsupported output_realizations={output_realizations}")

            self.y      = torch.tensor(y_np_filled_, dtype=torch.float32).permute(1, 0, 2, 3)  # (T,Cout,H,W)
            self.y_mask = torch.tensor(mask_y_np_,   dtype=torch.bool).permute(1, 0, 2, 3)     # (T,Cout,H,W)

            # print("TENSORS self.x", self.x.shape, "self.y", self.y.shape, "self.y_mask", self.y_mask.shape)
            assert self.y.shape == self.y_mask.shape, (self.y.shape, self.y_mask.shape)

            # Basic spatial + sample checks (channels may differ)
            assert self.x.shape[0]  == self.y.shape[0], (f"Mismatched dataset shape: x={self.x.shape}, y={self.y.shape}")  # same T
            assert self.x.shape[2:] == self.y.shape[2:], (f"Mismatched dataset shape: x={self.x.shape}, y={self.y.shape}") # same H,W

        else:
            if target_realization_avg and len(y_np.shape) == 5:
                y_np_filled = np.nanmean(y_np_filled, axis=2)          # average over R
                mask_y_np = np.any(mask_y_np, axis=2)                  # valid if any realization is valid

            # Uniform realizations: only works if one of input and target has R>1 and the other R=1
            if len(x_np_filled.shape) == 5: # C,T,R,H,W
                self.x = torch.tensor(x_np_filled, dtype=torch.float32).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3) # (T*R),C,H,W
                self.x_mask = torch.tensor(mask_x_np, dtype=torch.bool).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
            else: # C,T,H,W
                if len(y_np_filled.shape) == 5: # C,T,R,H,W
                    R = y_np_filled.shape[2]
                    x = torch.tensor(x_np_filled, dtype=torch.float32)      # C,T,H,W
                    x = x.unsqueeze(2).repeat(1, 1, R, 1, 1)                # C,T,R,H,W
                    self.x = x.flatten(1, 2).permute(1, 0, 2, 3)            # (T*R),C,H,W
                    x_mask = torch.tensor(mask_x_np, dtype=torch.bool)      # C,T,H,W
                    x_mask = x_mask.unsqueeze(2).repeat(1, 1, R, 1, 1)      # C,T,R,H,W
                    self.x_mask = x_mask.flatten(1, 2).permute(1, 0, 2, 3)  # (T*R),C,H,W
                else:
                    self.x = torch.tensor(x_np_filled, dtype=torch.float32).permute(1, 0, 2, 3) # T,C,H,W
                    self.x_mask = torch.tensor(mask_x_np, dtype=torch.bool).permute(1, 0, 2, 3)
            if len(y_np_filled.shape) == 5:
                self.y = torch.tensor(y_np_filled, dtype=torch.float32).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
                self.y_mask = torch.tensor(mask_y_np, dtype=torch.bool).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
            else:
                if len(x_np_filled.shape) == 5: # C,T,R,H,W
                    R = x_np_filled.shape[2]
                    y = torch.tensor(y_np_filled, dtype=torch.float32)      # C,T,H,W
                    y = y.unsqueeze(2).repeat(1, 1, R, 1, 1)                # C,T,R,H,W
                    self.y = y.flatten(1, 2).permute(1, 0, 2, 3)            # (T*R),C,H,W
                    y_mask = torch.tensor(mask_y_np, dtype=torch.bool)
                    y_mask = y_mask.unsqueeze(2).repeat(1, 1, R, 1, 1)
                    self.y_mask = y_mask.flatten(1, 2).permute(1, 0, 2, 3)
                else:
                    self.y = torch.tensor(y_np_filled, dtype=torch.float32).permute(1, 0, 2, 3)
                    self.y_mask = torch.tensor(mask_y_np, dtype=torch.bool).permute(1, 0, 2, 3)

            # print("self.x shape:", self.x.shape)
            # print("self.x_mask shape:", self.x_mask.shape)
            # print("self.y shape:", self.y.shape)
            # print("self.y_mask shape:", self.y_mask.shape)

            # Final checks
            assert self.x.numel() > 0, f"Empty torch dataset x: {self.x.shape}"
            assert self.y.numel() > 0, f"Empty torch dataset y: {self.y.shape}"
            assert self.x_mask.numel() > 0, f"Empty torch mask x_mask: {self.x_mask.shape}"
            assert self.y_mask.numel() > 0, f"Empty torch mask y_mask: {self.y_mask.shape}"

            # print("TENSORS self.x", self.x.shape, "self.y", self.y.shape, "self.y_mask", self.y_mask.shape)
            assert self.x.shape == self.x_mask.shape, f"Mismatched x tensor and its mask shapes: x={self.x.shape}, x_mask={self.x_mask.shape}"
            assert self.y.shape == self.y_mask.shape, f"Mismatched y tensor and its mask shapes: y={self.y.shape}, y_mask={self.y_mask.shape}"

            # self.x = torch.tensor(self.input_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
            # self.y = torch.tensor(self.target_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)

            assert self.x.shape == self.y.shape, f"Mismatched dataset shape: x={self.x.shape}, y={self.y.shape}"

    @staticmethod
    def _transpose_dims_ds_to_da(
        ds: xr.Dataset,
        excluded_vars: str | Sequence[str] | None = "_has_var"
    ) -> xr.DataArray:
        excluded_vars = excluded_vars or []
        vars = [v for v in ds.data_vars if v not in excluded_vars]
        da = ds[vars].to_array()

        required_dims = {
            'time': ds.earthml.guessed_dims.time,
            'y': ds.earthml.guessed_dims.latitude,
            'x': ds.earthml.guessed_dims.longitude,
        }
        realization_dim = da.earthml.guessed_dims.realization
        if set(required_dims.values()) - set(da.dims):
            raise ValueError(f"Unexpected dims: {da.dims}")

        if realization_dim in da.dims and da.ndim == 5:
            return da.transpose(
                "variable",
                required_dims['time'],
                realization_dim,
                required_dims['y'],
                required_dims['x'],
            )
        elif realization_dim not in da.dims and da.ndim == 4:
            return da.transpose(
                "variable",
                required_dims['time'],
                required_dims['y'],
                required_dims['x'],
            )
        else:
            return da

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]              # shapes: (C,H,W) or (C,T,H,W) if using time
        mx, my = self.x_mask[idx], self.y_mask[idx]

        if self.transform_x:
            x = self.transform_x(x, **self.transform_x_args)
        if self.transform_y:
            y = self.transform_y(y, **self.transform_y_args)

        mask = my           # use target mask
        # mask = mx & my    # combine input and target masks

        return x, y, mask