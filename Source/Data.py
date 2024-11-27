
from enum import Enum
from dataclasses import dataclass

class SegmentationType(Enum):
    Binary = 0
    Multiclass = 1

@dataclass
class BRect:
    x: int
    y: int
    w: int
    h: int