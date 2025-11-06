from dataclasses import asdict
from rich.table import Table
from pathlib import Path
from datetime import datetime
# Local imports
from .config import Combo
from .logging import Logger
from .experiment import ExperimentRegistry
from .dataclasses import TimeRange

class Launcher:
    def __init__ (self, tomlfn="earthml.toml"):
        self.combo = Combo(tomlfn)
        self.logger = Logger(
            Path(self.combo.combo_folder).joinpath("combo.log"),
            log_level=self.combo.cfg["global"]["log_level"]
        ).logger
        self.logger.debug(f"EarthML configuration file: {Path(tomlfn).resolve()}")
        self.logger.info(f'Init combo "{self.combo.name}" with {len(self.combo.runs)} runs')
        # self.test_input_exps = [ExperimentRegistry(r.source).get_class()(
        #     run_config=r, logger=self.logger, source_path=r.source_settings['input_path'], period=TimeRange(
        #         start=datetime.strptime(r.test['start_date'], '%Y-%m-%dT%H:%M:%S'),
        #         end=datetime.strptime(r.test['end_date'], '%Y-%m-%dT%H:%M:%S'),
        #         freq=r.source_settings['origin_frequency']
        #     )) for r in self.combo.runs]

    def generate_runs_rich_table (self) -> list[Table]:
        tables = []
        for r in self.combo.runs:
            table = Table(title=f"Run Configuration")
            table.add_column("Parameter", justify="left", style="cyan")
            table.add_column("Value", justify="left", style="green")
            for key, value in asdict(r).items():
                table.add_row(key, str(value))
            tables.append(table)
        return tables
