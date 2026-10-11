"""Small shared types for the bundle and scene models."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

type Vec2 = tuple[float, float]
type Vec3 = tuple[float, float, float]
type Row4 = Annotated[list[float], Field(min_length=4, max_length=4)]
# 4x4 matrix as a row-major nested list: m[row][col]. Translation is m[0..2][3].
type Mat4 = Annotated[list[Row4], Field(min_length=4, max_length=4)]

F64 = NDArray[np.float64]


class Strict(BaseModel):
    """Hand-edited or pipeline-owned files: unknown keys are typos, so reject them."""

    model_config = ConfigDict(extra="forbid")


class Tolerant(BaseModel):
    """Files written by other programs (the app, RoomPlan): ignore keys we don't read yet."""

    model_config = ConfigDict(extra="ignore")


def mat(m: Mat4) -> F64:
    return np.asarray(m, dtype=np.float64)


def to_mat4(a: F64) -> Mat4:
    return [[float(v) for v in row] for row in np.asarray(a, dtype=np.float64).reshape(4, 4)]
