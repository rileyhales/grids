from ._consts import ALL_STATS

__all__ = ['unknown_stat', 'unknown_open_file_object']


def unknown_stat(stat):
    return f'Unrecognized stat: {stat} - choose from {ALL_STATS}'


def unknown_open_file_object(obj_type):
    return f'Unrecognized opened file dataset: {obj_type}'
