import yaml
from pathlib import Path
from typing import Dict, Any
from src.validation import validate_config, ValidationError

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(f"Failed to parse YAML: {e}")
    
    if config is None:
        config = {}
    
    return validate_config(config)
