import os
import tensorflow as tf
import numpy as np


class TextDataset(object):
    """
    build a dataset of text, separator in line is space
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @staticmethod
    def get_raw_text_data(filepath):
        """
        load dataset into memory
        """
        with open(filepath, mode="rt", encoding="utf-8") as fp:
            return fp.read()


if __name__ == '__main__':
    data_path = ''
    # data_path = ''
