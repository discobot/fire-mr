from copy import copy
from itertools import groupby
from abc import abstractmethod

import logging
logger = logging.getLogger(__name__)


class Graph:
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MapMe(Graph):
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, records, **kwargs):
        for r in records:
            for r in self.mapper(r):
                yield r


class ReduceMe(Graph):
    def __init__(self, reducer, keys):
        self.reducer = reducer
        self.keys = keys

    def __call__(self, records, **kwargs):
        for key, group in groupby(records, key=lambda x: tuple(x[k] for k in self.keys)):
            for r in self.reducer(group):
                yield r


class JoinMe(Graph):
    def __init__(self, joiner, keys):
        self.keys = keys
        self.joiner = joiner

    def grouper(self, records):
        for key, group in groupby(records, key=self.get_key):
            yield key, group
        yield None, None

    def get_key(self, record):
        return tuple(record[k] for k in self.keys)

    def __call__(self, first_records, second_records, **kwargs):
        first_grouper = self.grouper(first_records)
        second_grouper = self.grouper(second_records)
        first_key, first_g = next(first_grouper)
        second_key, second_g = next(second_grouper)

        while first_key is not None and second_key is not None:
            if first_key < second_key:
                for r in self.joiner(first_g, []):
                    yield r
                first_key, first_g = next(first_grouper)
                continue

            if first_key == second_key:
                for r in self.joiner(first_g, second_g):
                    yield r
                first_key, first_g = next(first_grouper)
                second_key, second_g = next(second_grouper)
                continue

            if first_key > second_key:
                for r in self.joiner([], second_g):
                    yield r
                second_key, second_g = next(second_grouper)
                continue

        while first_key is not None:
            for r in self.joiner(first_g, []):
                yield r
            first_key, first_g = next(first_grouper)

        while second_key is not None:
            for r in self.joiner([], second_g):
                yield r
            second_key, second_g = next(second_grouper)


class AggregateMe(Graph):
    def __init__(self, aggregator, keys):
        self.aggregator = aggregator
        self.keys = keys

    def __call__(self, records, **kwargs):
        for key, group in groupby(records, key=lambda x: tuple(x[k] for k in self.keys)):
            state = {k: v for k, v in zip(self.keys, key)}
            for g in group:
                state = self.aggregator(g, state)
            yield state


class SortMe(Graph):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, records, **kwargs):
        for r in sorted(records, key=lambda x: tuple(x[k] for k in self.keys)):
            yield r


class SaveMe(Graph):
    def __init__(self, buffer):
        self.buffer = buffer

    def __call__(self, records, **kwargs):
        for r in records:
            self.buffer.append(r)
        return []

class ReadMe(Graph):
    def __init__(self, filename, parser):
        self.filename = filename
        self.parser = parser

    def __call__(self, **kwargs):
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterMe(Graph):
    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        for r in kwargs[self.name]:
            yield r


def dfs_run(start_node, visited):
    q = list()
    q.append(start_node)
    visited.add(start_node)
    processing_pass = []
    while q:
        top = q[-1]
        has_unvisited_childs = False
        for i in top.inputs:
            if i not in visited:
                q.append(i)
                visited.add(i)
                has_unvisited_childs = True
        if not has_unvisited_childs:
            processing_pass.append(top)
            q.pop()
    return processing_pass


class ParametrizedGraph:
    def __init__(self, params, graph):
        self.params = params
        self.graph = graph
        self.output = None

    @property
    def inputs(self):
        return self.params.values()

    def run(self, **kwargs):
        if not self.output:
            params = {k: v.run(**kwargs) for k, v in self.params.items()}
            self.output = list(self.graph(**params, **kwargs))

        return self.output

    @property
    def name(self):
        return self.graph.__class__.__name__


class FireMR:
    def __init__(self):
        self.graphs = []

    def read_from_file(self, filename, parser):
        graph = ReadMe(filename, parser)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({}, graph))
        return m

    def read_from_iter(self, it):
        graph = ReadIterMe(it)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({}, graph))
        return m

    def save(self, buffer):
        graph = SaveMe(buffer)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({"records": m.graphs[-1]}, graph))
        return m

    def map(self, mapper):
        graph = MapMe(mapper)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({"records": m.graphs[-1]}, graph))
        return m

    def sort(self, keys):
        graph = SortMe(keys)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({"records": m.graphs[-1]}, graph))
        return m

    def aggregate(self, aggregator, keys):
        graph = AggregateMe(aggregator, keys)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({"records": m.graphs[-1]}, graph))
        return m

    def reduce(self, reducer, keys):
        graph = ReduceMe(reducer, keys)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({"records": m.graphs[-1]}, graph))
        return m

    def join(self, joiner, join_graph, keys):
        graph = JoinMe(joiner, keys)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({
            "first_records": m.graphs[-1],
            "second_records": join_graph.graphs[-1]
        }, graph))
        return m

    def write(self, stream):
        graph = SaveMe(stream)
        m = FireMR()
        m.graphs = copy(self.graphs)
        m.graphs.append(ParametrizedGraph({"records": m.graphs[-1]}, graph))
        return m

    def get_path(self):
        visited = set()
        path = []
        for graph in self.graphs:
            if graph not in visited:
                path.extend(dfs_run(graph, visited))
        return path

    def run(self, verbose=True, **kwargs):
        path = self.get_path()
        if verbose:
            logger.error("Execution path: {}".format(", ".join(p.name for p in path)))

        for p in self.get_path():
            p.output=None

        for p in self.get_path():
            p.run(**kwargs)

        return list(p.output)
