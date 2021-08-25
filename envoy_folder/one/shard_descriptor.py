# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Market shard descriptor."""

import re
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class NextWordShardDescriptor(ShardDescriptor):
    """"Data - any text."""

    def __init__(self, title: str = '', author: str = '') -> None:
        """Initialize NextWordShardDescriptor."""
        super().__init__()

        self.title = title
        self.author = author
        self.dataset_dir = list(Path.cwd().rglob(f'{title}.txt'))[0]
        self.data = self.load_data(self.dataset_dir)  # list of words
        self.X, self.y = self.get_sequences(self.data)

    def __len__(self):
        """Number of sequences."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.X[index], self.y[index]

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return self.X[0].shape

    @property
    def target_shape(self):
        """Return the target shape info."""
        return self.y[0].shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return f'Dataset from {self.title} by {self.author}'

    @staticmethod
    def load_data(path):
        """Load text file, return list of words."""
        file = open(path, 'r', encoding='utf8').read()
        data = re.findall(r'[a-z]+', file.lower())
        return data

    @staticmethod
    def get_sequences(data):
        """
        Transform words to sequences.

        To make vocab and clean it:
            if not spacy.util.is_package('en_core_web_sm'):
                spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
            clean_vocab_list = [word for word in nlp.vocab.strings if re.fullmatch(r'[a-z]+', word)]
        """

        # spacy en_core_web_sm vocab_size = 48904
        clean_vocab_list = open('vocab.txt', encoding='utf-8').read().split('\n')
        vocab = {word: idx for idx, word in enumerate(clean_vocab_list)}

        x_seq = []
        y_seq = []
        for i in range(len(data) - 3):
            x = data[i:i + 3]  # make 3-grams
            y = data[i + 3]
            cur_x = [vocab[word] for word in x if word in vocab]
            if len(cur_x) == 3 and y in vocab:
                x_seq.append(cur_x)
                y_seq.append(vocab[y])

        x_seq = np.array(x_seq)
        y_seq = to_categorical(y_seq, num_classes=len(clean_vocab_list))
        return x_seq, y_seq
