import yaml
from pathlib import Path

class Config:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(self.path, "r") as f:
            self.data = yaml.safe_load(f)
        self._validate()

    def _validate(self):
        required = ["model", "input", "output", "runtime"]
        for k in required:
            if k not in self.data:
                raise ValueError(f"Missing config section: {k}")

    def get(self, *keys, default=None):
        ref = self.data
        for k in keys:
            if k not in ref:
                return default
            ref = ref[k]
        return ref
