import os
from typing import Any, Optional
from pymatgen.core.composition import Composition

def sanitize_path(path: str) -> str:
    return path.strip().replace('\n', '').replace('`', '')

def safe_composition_conversion(x: Any) -> Optional[Composition]:
    if not isinstance(x, str):
        return None
    try:
        return Composition(x)
    except Exception:
        return None

def ensure_directory_exists(file_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True) 