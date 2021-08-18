import warnings
import numpy as np


def _map_coords_to_slice(vals: np.ndarray, coord_min: int, coord_max: int, var: str) -> int or slice:
    # reduce the number of dimensions on the coordinate variable if applicable
    if vals.ndim < 2:
        pass
    if vals.ndim == 2:
        if vals[0, 0] == vals[0, 1]:
            vals = vals[:, 0]
        elif vals[0, 0] == vals[1, 0]:
            vals = vals[0, :]
        else:
            raise RuntimeError("A 2D coordinate variable had non-uniform values and couldn't be reduced")
    elif vals.ndim > 2:
        raise RuntimeError(f"Coordinate variable should be 1-dimensional, was {vals.ndim}-dimensional")

    min_val = vals.min()
    max_val = vals.max()
    index1 = _map_coord_to_index(vals, min_val, coord_min, max_val, var)
    index2 = index1 if coord_max == coord_min else _map_coord_to_index(vals, min_val, coord_max, max_val, var)
    return _map_indices_to_slice(index1, index2)


def _map_coord_to_index(vals: np.ndarray, min_val: float, val: float, max_val: float, var: str) -> int or None:
    if val is None:
        return None
    if max_val >= val >= min_val:
        index = (np.abs(vals - val)).argmin()
    else:
        warnings.warn(f'Value ({val}) is outside min/max range ({min_val}, {max_val}) for dimension ({var})')
        if val >= max_val:
            warnings.warn(f'Defaulting to largest value: {max_val}')
            index = (np.abs(vals - max_val)).argmin()
        else:
            warnings.warn(f'Defaulting to smallest value: {min_val}')
            index = (np.abs(vals - min_val)).argmin()
    return index


def _map_indices_to_slice(index1: int or None, index2: int or None) -> int or slice:
    # if either or both are None
    if index1 is None and index2 is None:
        return slice(None)
    elif index1 is None:
        return slice(index2)
    elif index2 is None:
        return slice(index1, -1)

    # if both are integers
    elif index1 == index2:
        return index1
    elif index1 < index2:
        return slice(index1, index2)
    else:
        return slice(index2, index1)