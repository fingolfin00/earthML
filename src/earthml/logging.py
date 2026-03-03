from dataclasses import dataclass
import datetime
from rich import print

@dataclass(frozen=True)
class RunKey:
    var_exp: str
    region: str
    experiment_type: str
    var_fc_key: str
    var_an_key: str
    loss_suf: str
    lt_fc_value: int | float
    lt_fc_unit: str
    lt_value: int | float
    lt_unit: str
    train_in_start: datetime
    train_in_end: datetime
    train_tar_start: datetime
    train_tar_end: datetime

    def short(self) -> str:
        # Compact, stable, grep-friendly
        return (
            f"{self.experiment_type}|{self.var_exp}|{self.region}|"
            f"{self.loss_suf}|fc{self.lt_fc_value}{self.lt_fc_unit}|"
            f"lt{self.lt_value}{self.lt_unit}|"
            f"in{self.train_in_start:%Y%m}-{self.train_in_end:%Y%m}|"
            f"tar{self.train_tar_start:%Y%m}-{self.train_tar_end:%Y%m}"
        )

def log_event(kind: str, run_key: RunKey | None = None, msg: str = "", **kv):
    # kind examples: PLAN, RUN, OK, RETRY, FAIL, SKIP, SUMMARY
    base = f"[{kind}]"
    if run_key is not None:
        base += f" {run_key.short()}"
    if msg:
        base += f" | {msg}"
    if kv:
        extra = " ".join(f"{k}={v}" for k, v in kv.items())
        base += f" | {extra}"
    print(base)
