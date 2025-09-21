import math

import numpy as np


def irc_to_xyz(
    coord_irc: tuple[int, int, int],
    origin_xyz: np.ndarray,
    vx_size_xyz: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """IRC = (index, row, col) = (z, y, x)."""
    coord = np.array([coord_irc[2], coord_irc[1], coord_irc[0]])
    coord = vx_size_xyz * coord
    coord = direction @ coord
    coord = coord + origin_xyz

    return coord


def xyz_to_irc(
    coord_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    vx_size_xyz: np.ndarray,
    direction: np.ndarray,
) -> tuple[int, int, int]:
    coord = np.array([*coord_xyz])
    coord = coord - origin_xyz
    coord = np.linalg.inv(direction) @ coord
    coord = coord / vx_size_xyz

    return np.array([coord[2], coord[1], coord[0]])


def coord_distance(
    first_coord: tuple[float, float, float], second_coord: tuple[float, float, float]
) -> float:
    return math.sqrt(
        (first_coord[0] - second_coord[0]) ** 2
        + (first_coord[1] - second_coord[1]) ** 2
        + (first_coord[2] - second_coord[2]) ** 2
    )
