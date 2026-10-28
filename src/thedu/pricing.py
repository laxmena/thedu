"""Reference hosted prices. Local runs are priced like hosted ones; nothing is free."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from thedu.paths import ROOT


class Rate(BaseModel):
    input_per_mtok_usd: float
    output_per_mtok_usd: float


class Pricing(BaseModel):
    fetched: str
    source: str
    models: dict[str, Rate]

    def rate(self, model: str) -> Rate:
        if model not in self.models:
            raise KeyError(
                f"no reference price for {model!r}. Add the hosted price of a model of the same "
                "size to configs/pricing.yaml (with date and source); local runs are never free."
            )
        return self.models[model]


def load_pricing(path: str | Path) -> Pricing:
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return Pricing.model_validate(yaml.safe_load(p.read_text()))
