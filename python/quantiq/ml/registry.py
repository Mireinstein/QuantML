"""Loads whichever model train.py most recently selected, using the
metadata it wrote (model type + which file to load) rather than every
caller having to know both model classes and pick the right one."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .model import SklearnSignalModel, TorchSignalModel

HERE = Path(__file__).resolve().parent
METADATA_PATH = HERE / "model_metadata.json"


class ModelNotTrainedError(RuntimeError):
    pass


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise ModelNotTrainedError(f"No trained model at {METADATA_PATH} -- run `python -m quantiq.ml.train` first.")
    return json.loads(METADATA_PATH.read_text())


def load_best_model() -> Union[SklearnSignalModel, TorchSignalModel]:
    metadata = load_metadata()
    model_path = HERE / metadata["model_file"]
    if not model_path.exists():
        raise ModelNotTrainedError(f"model_metadata.json points at {model_path}, but that file is missing.")

    if metadata["model_type"] == "gru":
        return TorchSignalModel.load(model_path)
    return SklearnSignalModel.load(model_path)
