import logging

import numpy as np

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Data"


class Data:  # TODO: add table class
    tag = tag

    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return {
            **self._data,
            "type": self.__class__.__name__,
        }


class Graph(Data):
    tag = "Graph"

    def __init__(self, data={}):
        self._nodes = data.get("nodes", {})
        self._edges = data.get("edges", {})
        self._successors = data.get("successors", {})
        self._predecessors = data.get("predecessors", {})
        super().__init__(data=self.to_data())

    def add_node(self, node, **attr):
        if node not in self._nodes:
            self._nodes[node] = attr
            self._successors[node] = set()
            self._predecessors[node] = set()
        else:
            self._nodes[node].update(attr)

    def add_edge(self, src, dst, **attr):
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)

        self._edges[(src, dst)] = attr
        self._successors[src].add(dst)
        self._predecessors[dst].add(src)

    @property
    def nodes(self):
        return self._nodes

    def edges(self, data=False):
        if data:
            return list(self._edges.items())
        return list(self._edges.keys())

    def to_data(self):
        return {
            "nodes": self._nodes,
            "edges": self._edges,
            "successors": self._successors,
            "predecessors": self._predecessors,
        }


class Histogram(Data):
    tag = "Histogram"

    def __init__(self, data, bins=64):
        self._shape = "generic"
        if len(data) == 2 and not isinstance(bins, int):
            # TODO: support non-uniform bins
            logger.debug(
                f"{tag}: using pre-set bins from data; bins need to have uniform intervals"
            )
            d = data[0].tolist() if hasattr(data[0], "tolist") else data[0]
            b = data[1].tolist() if hasattr(data[1], "tolist") else data[1]
            if len(d) + 1 != len(b):
                logger.critical(
                    f"{self.tag}: data and length must be the same length: force proceeding"
                )
            else:
                self._shape = "uniform"
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            d, b = np.histogram(data, bins=bins)
            d, b = d.tolist(), b.tolist()
        else:
            logger.critical(
                f"{self.tag}: data must be a list or numpy array: force proceeding with an empty histogram"
            )
            d, b = [0], [0, 1]

        self._freq = d
        self._bins = b
        super().__init__(data=self.to_data())

    def to_data(self):
        return {
            "freq": self._freq,
            "bins": self._bins,
            "shape": self._shape,
        }
