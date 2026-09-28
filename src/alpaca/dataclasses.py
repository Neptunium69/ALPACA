from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class FCReturn():
    function: str
    data: Any | None

@dataclass
class FCExecute():
    function: str
    arguments: dict[str:Any]

@dataclass
class Function():
    name:str
    arguments: dict[str,tuple[str,bool]]
