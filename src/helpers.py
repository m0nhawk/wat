import pandas as pd


def date_formatter(x, pos):
    """

    :param x:
    :param pos:
    :return:
    """
    return pd.to_datetime(x, unit='s').strftime('%H:%M:%S')
