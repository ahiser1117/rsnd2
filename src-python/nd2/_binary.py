from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(frozen=True)
class BinaryLayer:
    name: str
    data: Sequence[object | None]
    comp_name: str | None = None
    comp_order: int | None = None
    color: int | None = None
    color_mode: int | None = None
    state: int | None = None
    file_tag: str | None = None
    layer_id: int | None = None

    def __getitem__(self, index: int) -> object | None:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def asarray(self):
        import numpy as np

        return np.asarray(self.data)

    def __array__(self):
        return self.asarray()


class BinaryLayers(tuple[BinaryLayer, ...]):
    def __new__(cls, layers: Sequence[BinaryLayer] = ()) -> "BinaryLayers":
        return super().__new__(cls, layers)

    def __iter__(self) -> Iterator[BinaryLayer]:
        return super().__iter__()

    def asarray(self):
        import numpy as np

        return np.asarray([layer.asarray() for layer in self])

    def __array__(self):
        return self.asarray()
