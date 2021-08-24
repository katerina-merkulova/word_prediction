# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Market shard descriptor."""

import re
from pathlib import Path

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class NextWordShardDescriptor(ShardDescriptor):
    """"Data - any text."""

    def __init__(self, title: str = '', author: str = '') -> None:
        """Initialize NextWordShardDescriptor."""
        super().__init__()

        self.title = title
        self.author = author
        self.dataset_dir = list(Path.cwd().rglob(f'{title}.txt'))[0]
        self.data = self.load(self.dataset_dir)
        self.tokenizer, self.sequences = self.tokenize(self.data)
        self.X, self.y = self.transform(self.sequences, num_classes=len(self))

    def __len__(self):
        """Length of dataset."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.X[index], self.y[index]

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['1']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return f'Dataset from {self.title} by {self.author}'

    @staticmethod
    def load(path):
        """Load text file, return list of words."""
        file = open(path, 'r', encoding='utf8').read()
        data = re.findall(r'\w+', file)
        return data
