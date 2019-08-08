import collections

import numpy as np


class Collector(object):
    def __init__(self):
        self.data = collections.defaultdict(list)

    def keys(self):
        return self.data.keys()

    def add(self, key, value):
        if not np.isnan(value):
            self.data[key].append(value)

    def __getitem__(self, key):
        return self.data.get(key, [])
