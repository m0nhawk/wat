import pandas as pd


def get_formatted_time(time, date_format):
    """

    :param time:
    :param date_format:
    :return:
    """
    if date_format:
        time = pd.to_datetime(time, format=date_format)
    else:
        time = pd.to_datetime(time, unit='s')
    return time


def date_formatter(x, pos):
    """

    :param x:
    :param pos:
    :return:
    """
    return pd.to_datetime(x, unit='s').strftime('%H:%M:%S')
