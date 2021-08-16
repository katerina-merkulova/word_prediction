# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Market shard descriptor."""

import pickle
import re
from logging import getLogger

logger = getLogger(__name__)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


DATAPATH = list(Path.cwd().rglob('data'))[0]    # parent directory of project


class NextWordShardDescriptor(ShardDescriptor):
    """
        Dataset: http://www.gutenberg.org/cache/epub/5200/pg5200.txt
        Remove all the unnecessary data and label it as Metamorphosis.
        The starting and ending lines should be as follows.
        The First Line: One morning, when Gregor Samsa woke from troubled dreams, he found
        The Last Line:  first to get up and stretch out her young body.
    """

    def __init__(self, data_folder: str = 'Market',
                 rank_worldsize: str = '1,1') -> None:
        """Initialize NextWordShardDescriptor."""
        super().__init__()

         # Settings for sharding the dataset
        self.rank_worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        self.dataset_dir = Path(DATAPATH)
        self.data = self.load(self.dataset_dir)
        self.tokenizer, self.sequences = self.tokenizer(self.data)
        self.X, self.y = self.transform(self.sequences)

    def __len__(self):        
        return len(self.tokenizer.word_index) + 1

    def __getitem__(self, index: int): # todo
        """Return a item by the index."""
        return img, pid

    @property
    def sample_shape(self): # todo
        """Return the sample shape info."""
        return ['64', '128', '3']

    @property
    def target_shape(self): # todo
        """Return the target shape info."""
        return ['1501']

    @property
    def dataset_description(self) -> str: # todo
        """Return the dataset description."""
        return f'Market dataset, shard number {self.rank_worldsize[0]}' \
               f' out of {self.rank_worldsize[1]}'

    @staticmethod
    def load(path):
        file = open(path / 'metamorphosis.txt', 'r', encoding='utf8').read()
        data = re.findall('\w+', file)
        return data

    @staticmethod
    def tokenize(data):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([data])
        # saving the tokenizer for predict function.
        pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))
        sequence_data = tokenizer.texts_to_sequences([data])[0]

        sequences = []
        for i in range(1, len(sequence_data)):
            words = sequence_data[i-1:i+1]
            sequences.append(words)
        sequences = np.array(sequences)
        return tokenizer, sequences

    @staticmethod
    def transform(sequences):
        X, y = [], []
        for i in sequences:
            X.append(i[0])
            y.append(i[1])
        X = np.array(X)
        y = np.array(y)
        y = to_categorical(y, num_classes=len(self))
        return X, y