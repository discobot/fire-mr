from hw2.lib import mr
from hw2.lib import operations


def test_initialization():
    mr.FireMR()


def test_add_read_node():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    m = mr.FireMR().read_from_iter(rows)
    classes = [g.name for g in m.get_path()]
    assert classes == ['ReadIterMe']


def test_add_read_and_save_node():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    etalon = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    buffer = []
    m = mr.FireMR().read_from_iter(rows).save(buffer)
    classes = [g.name for g in m.get_path()]
    assert classes == ['ReadIterMe', 'SaveMe']
    m.run()
    assert buffer == etalon


def test_add_read_and_map_and_save_node():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    etalon = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    buffer = []
    m = mr.FireMR().read_from_iter(rows).map(operations.Dummy()).save(buffer)
    classes = [g.name for g in m.get_path()]
    assert classes == ['ReadIterMe', 'MapMe', 'SaveMe']
    m.run()
    assert buffer == etalon


def test_add_read_and_aggregate_and_save_node():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5},  {'a': 1, 'b': 2}, {'a': 2, 'b': 2}, {'a': 2, 'b': 5}, {'a': 3, 'b': 1}]
    etalon = [{'a': 1, 'b': 8}, {'a': 2, 'b': 7}, {'a': 3, 'b': 1}]
    buffer = []
    m = mr.FireMR().read_from_iter(rows).aggregate(operations.Sum('b'), ['a']).save(buffer)
    classes = [g.name for g in m.get_path()]
    assert classes == ['ReadIterMe', 'AggregateMe', 'SaveMe']
    m.run()
    assert buffer == etalon


def test_add_read_and_reduce_and_save_node():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5}, {'a': 1, 'b': 2}, {'a': 2, 'b': 2}, {'a': 2, 'b': 5}, {'a': 3, 'b': 1}]
    etalon = [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}, {'a': 3, 'b': 1}]
    buffer = []
    m = mr.FireMR().read_from_iter(rows).reduce(operations.FirstReducer(), ['a']).save(buffer)
    classes = [g.name for g in m.get_path()]
    assert classes == ['ReadIterMe', 'ReduceMe', 'SaveMe']
    m.run()
    assert buffer == etalon


def test_add_read_and_sort_and_join_and_save_node():
    rows_a = [
        {'word': 'hello', 'count': 1}, {'word': 'hello', 'count': 10},
        {'word': 'hi', 'count': 15}, {'word': 'hi', 'count': 20}
    ]

    rows_b = [{'word': 'hello', 'other': 11}, {'word': 'hello', 'other': 12},
              {'word': 'my', 'other': 15}, {'word': 'my', 'other': 20}]

    etalon = [
        {'word': 'hello', 'count': 1, 'other': 11}, {'word': 'hello', 'count': 1, 'other': 12},
        {'word': 'hello', 'count': 10, 'other': 11}, {'word': 'hello', 'count': 10, 'other': 12},
        {'word': 'hi', 'count': 15}, {'word': 'hi', 'count': 20},
        {'word': 'my', 'other': 15}, {'word': 'my', 'other': 20}
    ]

    buffer = []
    a = mr.FireMR().read_from_iter(rows_a).sort(['word'])
    m = mr.FireMR().read_from_iter(rows_b).sort(['word']).join(operations.OuterJoiner(), a, ['word']).save(buffer)
    classes = [g.name for g in m.get_path()]

    assert classes == ['ReadIterMe', 'SortMe', 'ReadIterMe', 'SortMe', 'JoinMe', 'SaveMe']
    m.run()

    assert sorted(buffer, key=lambda x: (x['word'], x.get('count', 0), x.get('other', 0))) == etalon


def test_add_read_and_sort_and_save_node():
    rows = [{'a': 3, 'b': 2}, {'a': 4, 'b': 1}]
    etalon = [{'a': 4, 'b': 1}, {'a': 3, 'b': 2}]
    buffer = []
    m = mr.FireMR().read_from_iter(rows).sort(['b']).save(buffer)
    classes = [g.name for g in m.get_path()]
    assert classes == ['ReadIterMe', 'SortMe', 'SaveMe']
    m.run()
    assert buffer == etalon


def test_map_graph():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    etalon = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    m = mr.MapMe(operations.Dummy())
    result = m(rows)
    assert list(result) == etalon


def test_sort_graph():
    rows = [{'a': 3, 'b': 2}, {'a': 4, 'b': 1}]
    etalon = [{'a': 4, 'b': 1}, {'a': 3, 'b': 2}]
    m = mr.SortMe(keys=['b'])
    result = m(rows)
    assert list(result) == etalon


def test_join_graph():
    rows_a = [
        {'word': 'hello', 'count': 1}, {'word': 'hello', 'count': 10},
        {'word': 'hi', 'count': 15}, {'word': 'hi', 'count': 20}
    ]

    rows_b = [
        {'word': 'hello', 'other': 11}, {'word': 'hello', 'other': 12},
        {'word': 'my', 'other': 15}, {'word': 'my', 'other': 20}
    ]

    etalon = [
        {'word': 'hello', 'count': 1, 'other': 11}, {'word': 'hello', 'count': 1, 'other': 12},
        {'word': 'hello', 'count': 10, 'other': 11}, {'word': 'hello', 'count': 10, 'other': 12},
        {'word': 'hi', 'count': 15}, {'word': 'hi', 'count': 20},
        {'word': 'my', 'other': 15}, {'word': 'my', 'other': 20}
    ]

    m = mr.JoinMe(operations.OuterJoiner(), keys=['word'])
    result = m(rows_a, rows_b)
    assert sorted(list(result), key=lambda x: (x['word'], x.get('count', 0), x.get('other', 0))) == etalon


def test_aggregate_graph():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5},  {'a': 1, 'b': 2}, {'a': 2, 'b': 2}, {'a': 2, 'b': 5}, {'a': 3, 'b': 1}]
    etalon = [{'a': 1, 'b': 8}, {'a': 2, 'b': 7}, {'a': 3, 'b': 1}]
    m = mr.AggregateMe(operations.Sum('b'), keys=['a'])
    result = m(rows)
    assert list(result) == etalon


def test_reduce_graph():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5},  {'a': 1, 'b': 2}, {'a': 2, 'b': 2}, {'a': 2, 'b': 5}, {'a': 3, 'b': 1}]
    etalon = [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}, {'a': 3, 'b': 1}]
    m = mr.ReduceMe(operations.FirstReducer(), keys=['a'])
    result = m(rows)
    assert list(result) == etalon


def test_read_iter_graph():
    it = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5}]
    etalon = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5}]
    m = mr.ReadIterMe(it)
    result = m()
    assert list(result) == etalon


def test_save_graph():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5}]
    etalon = [{'a': 1, 'b': 1}, {'a': 1, 'b': 5}]
    buffer = []
    m = mr.SaveMe(buffer)
    result = m(rows)
    assert list(result) == []
    assert buffer == etalon
