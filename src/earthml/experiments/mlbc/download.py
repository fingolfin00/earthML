# from typing import Sequence, Dict, Tuple, Literal
# from dataclasses import asdict

# import time, multiprocessing, joblib
# from pathlib import Path

# from rich import print
# from rich.console import Console

# import numpy as np
# import pandas as pd
# import xarray as xr
# from zarr.codecs import BloscCodec

# from .dataclasses import MLBCExperimentDataset, MLBCExperimentConfig, MLBCExperimentDatasetRole

# from ...sources import build_source, BaseSource, DataSource, XarrayLocalSourceConfig

# from ...misc import Table


# class MLBCDownload:
#     def __init__(
#         self,
#         config: MLBCExperimentConfig,
#     ):
#         self.config = config
#         self.rich_console = Console()

#         # Get test variable list,
#         self.test_var_list = self._get_var_list(self.config.test_dataset)
#         self.train_var_list = self._get_var_list(self.config.train_dataset)

#         self.consolidated_zarr = False

#         # Init source data objects and save original datasets
#         self.source_train_data = self._init_source_data(self.config.train_dataset, "train")
#         self.source_test_data  = self._init_source_data(self.config.test_dataset,  "test")

#         # Generate xarray datsets
#         self.train_input_ds, self.train_target_ds = self._generate_xarray_ds(self.source_train_data, self.config.train_dataset, 'Train') # capital T just for logging
#         self.test_input_ds, self.test_target_ds = self._generate_xarray_ds(self.source_test_data, self.config.test_dataset, 'Test')