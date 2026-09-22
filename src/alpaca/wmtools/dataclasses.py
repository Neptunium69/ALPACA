from dataclasses import dataclass,field
from typing import Any

@dataclass
class WMQuery:
    query: str
    filters: dict[str, Any] = field(default_factory=dict)
