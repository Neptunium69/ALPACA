from dataclasses import dataclass
from typing import Any

@dataclass
class VAReturn():
    function: str
    data: Any | None

@dataclass
class VAExecute():
    function: str
    arguments: dict[str,Any]

@dataclass
class Function():
    name:str
    arguments: dict[str,tuple[str,bool]]
