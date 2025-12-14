
from enum import Enum
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Dict, List, Tuple

class SegmentationType(Enum):
    BINARY = 0
    MULTICLASS = 1


@dataclass
class BRect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Bbox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class LineData:
    contour: List
    bbox: Bbox
    center: Tuple[int, int]


@dataclass
class PerigPrediction:
    images: NDArray
    lines: NDArray
    captions: NDArray
    margins: NDArray


@dataclass
class LayoutData:
    images: List[Bbox]
    text_bboxes: List[Bbox]
    lines: List[LineData]
    captions: List[Bbox]
    margins: List[Bbox]
    predictions: Dict
