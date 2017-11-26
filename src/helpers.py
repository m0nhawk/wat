import pandas as pd


def get_formatted_time(time, date_format=None):
    """

    :param time: list or 1-d ndarray with datetimes to convert
    :param date_format: None for Unix timestamp, string with format otherwise
    :return: list or Series with datetime's
    """
    if date_format:
        time = pd.to_datetime(time, format=date_format)
    else:
        time = pd.to_datetime(time, unit='s')
    return time


def date_formatter(x, _):
    """
    `matplotlib` helper to convert timestamp to HH:MM:SS format

    :param x: Unix timestamp to convert
    :param _: Unused, but required function argument
    :return: String, with datetime in 'HH:MM:SS' format
    """
    return pd.to_datetime(x, unit='s').strftime('%H:%M:%S')
