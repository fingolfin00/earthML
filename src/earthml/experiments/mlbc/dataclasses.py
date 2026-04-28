from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Literal, Any

from pathlib import Path

import xarray as xr

from ...sources import DataSource, EarthkitSourceConfig, JunoLocalSourceConfig, CopernicusmarineSourceConfig, XarrayLocalSourceConfig
from .registry import MLBCExperimentType, MLBCExperimentName, MLBCExperimentDatasetRole

PreprocessFn = Callable[[xr.Dataset, xr.Dataset], Tuple[xr.Dataset, xr.Dataset]]
DatasetRebuildSelector = Literal[
    "train_input",
    "train_target",
    "test_input",
    "test_target",
    "train",
    "test",
    "input",
    "target",
    "all",
]

# Dataclasses
@dataclass(frozen=True)
class MLBCExperimentLauncherConfig:
    type        : MLBCExperimentType
    name        : MLBCExperimentName
    root_path   : str | Path
    run_name_suffix: str

@dataclass(frozen=True)
class MLBCExperimentDataset:
    role            : MLBCExperimentDatasetRole
    datasource      : DataSource | List[DataSource]
    save            : bool = False
    source_configs  : EarthkitSourceConfig | JunoLocalSourceConfig | CopernicusmarineSourceConfig | XarrayLocalSourceConfig | None = None

@dataclass(frozen=True)
class MLBCNeuralNet:
    """
    MLBC neural hyperparameters. Provides common ML knobs
    """
    # Reproducibility
    seed                    : int   = 42
    # Net name
    name                    : str   = "SmaAt_UNet"
    # Net config
    supervised              : bool  = True
    n_channels              : int   = 1 # NN input C
    n_classes               : int   = 1 # NN output C
    extra_net_args          : dict[str, Any] = field(default_factory=dict)
    # Net hyperparams
    learning_rate           : float = 1e-3
    batch_size              : int   = 32
    epochs                  : int   = 50
    # Loss
    loss                    : str   = "MSELoss"
    loss_params             : dict[str, dict[str, Any]] = field(default_factory=lambda: {"net": {}, "loss": {}})
    # Other
    norm_strategy           : str   = "BatchNorm2d"
    trainer_precision       : Optional[str] = None
    cuda_alloc_conf         : Optional[str] = "expandable_segments:True"
    train_percent           : float = 0.9
    earlystopping_patience  : int   = 30
    accumulate_grad_batches : int   = 2


@dataclass
class MLBCExperimentConfig:
    # Name and folders
    name                        : str
    work_path                   : str | Path
    # download_path               : str | Path
    # NN and hyperparams
    net                         : MLBCNeuralNet
    # Dataset
    train_dataset               : MLBCExperimentDataset | List[MLBCExperimentDataset]
    test_dataset                : MLBCExperimentDataset | List[MLBCExperimentDataset]
    # Optional # TODO create a dedicated dataclass
    run_name_suffix             : Optional[str]                    = None              # full launcher suffix, e.g. "_mseloss_debug"
    anomaly                     : Optional[bool]                    = False             # train on anomaly wrt input and target climatologies
    inpaint_nan                 : Optional[bool]                    = False             # whether to inpaint nan values in input and target datasets (after loading, before torch dataset generation)
    per_epoch_resplit           : Optional[bool]                    = False             # resplit train and validation sets per epoch with different seed
    torch_preprocess_fn         : Optional[PreprocessFn]            = None              # called after Xarray dataset loading, before torch dataset generation
    target_realization_avg      : Optional[bool]                    = False             # whether to average over target realizations when loading target data
    realization_as_channel      : Optional[bool]                    = False             # whether to use realization a channel dimension
    output_realizations         : Optional[Literal[
                                    "deterministic", "ensemble"]]   = "deterministic"   # deterministic -> output R = 1, ensemble -> output R = input R
    skip_train_test_plots       : Optional[bool]                    = False             # whether to skip train/test diagnostic plot generation
    external_mask_path          : Optional[str | Path]              = None              # optional external mask file/store to intersect with common input/target validity
    external_mask_variable      : Optional[str]                     = None              # optional variable selection within external mask dataset
    trim_invalid_border_lines   : Optional[bool]                    = False             # whether to crop fully invalid outer lat/lon rows or columns before torch dataset creation
    force_rebuild_dataset       : Optional[bool | DatasetRebuildSelector] = False       # False to reuse saved stores, else rebuild matching dataset stores
    dataset_cache_enabled       : Optional[bool]                    = False             # whether to reuse shared cached source-role datasets across experiments
    dataset_cache_root          : Optional[str | Path]              = None              # root folder for shared dataset cache metadata and stores
    pass_mask_as_input_extra_channel: Optional[bool]                = False             # C -> 2C
