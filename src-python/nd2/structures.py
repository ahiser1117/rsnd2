from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Attributes:
    bitsPerComponentInMemory: int
    bitsPerComponentSignificant: int
    componentCount: int
    heightPx: int
    pixelDataType: str
    sequenceCount: int
    widthBytes: int
    widthPx: int
    compressionLevel: int | None = None
    compressionType: str | None = None
    tileHeightPx: int | None = None
    tileWidthPx: int | None = None
    channelCount: int | None = None


@dataclass(frozen=True)
class Contents:
    channelCount: int
    frameCount: int


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int
    a: float = 1.0


@dataclass(frozen=True)
class ChannelMeta:
    name: str
    index: int
    color: Color | None = None
    emissionLambdaNm: float | None = None
    excitationLambdaNm: float | None = None


@dataclass(frozen=True)
class Channel:
    channel: ChannelMeta
    loops: Any = None
    microscope: Any = None
    volume: Any = None


@dataclass(frozen=True)
class Metadata:
    contents: Contents
    channels: list[Channel] = field(default_factory=list)


@dataclass(frozen=True)
class FrameMetadata:
    contents: Contents
    channels: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class TextInfo:
    description: str | None = None
    capturing: str | None = None
    date: str | None = None
    optics: str | None = None


@dataclass(frozen=True)
class TimeStamp:
    absoluteJulianDayNumber: float | None = None
    relativeTimeMs: float | None = None


@dataclass(frozen=True)
class StagePosition:
    x: float | None = None
    y: float | None = None
    z: float | None = None


@dataclass(frozen=True)
class Position:
    stagePositionUm: StagePosition | list[float] | None = None
    pfsOffset: float | None = None
    name: str | None = None


@dataclass(frozen=True)
class XYPosLoopParams:
    points: list[Position] = field(default_factory=list)
    isSettingZ: bool | None = None


@dataclass(frozen=True)
class XYPosLoop:
    count: int
    nestingLevel: int
    parameters: XYPosLoopParams
    type: str = "XYPosLoop"


@dataclass(frozen=True)
class ROI:
    id: int
    info: Any = None
    guid: str | None = None
    animParams: list[Any] = field(default_factory=list)


ExpLoop = Any
