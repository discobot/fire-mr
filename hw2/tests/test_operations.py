import pytest
from hw2.lib import operations


def test_filter_punctuations():
    row = {'a': 1, 'b': "a, b"}
    etalon = [{'a': 1, 'b': "a b"}]
    op = operations.FilterPunctuation('b')
    assert list(op(row)) == etalon


def test_lower_case():
    row = {'a': 1, 'b': "HeLLo worlD"}
    etalon = [{'a': 1, 'b': "hello world"}]
    op = operations.LowerCase('b')
    assert list(op(row)) == etalon


def test_split():
    row = {'a': 1, 'b': "a b"}
    etalon = [{'a': 1, 'b': "a"}, {'a': 1, 'b': "b"}]
    op = operations.Split('b', ' ')
    assert list(op(row)) == etalon


def test_dummy():
    row = {'a': 1, 'b': "a"}
    etalon = [{'a': 1, 'b': "a"}]
    op = operations.Dummy()
    assert list(op(row)) == etalon


def test_sum():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    etalon = {'a': 1, 'b': 3}
    op = operations.Sum('b')
    state = {'a': 1}
    for r in rows:
        state = op(r, state)
    assert state == etalon


def test_count():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    etalon = {'a': 1, 'b': 2}
    op = operations.Count('b')
    state = {'a': 1}
    for r in rows:
        state = op(r, state)
    assert state == etalon


def test_first_reducer():
    rows = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    etalon = [{'a': 1, 'b': 1}]
    op = operations.FirstReducer()
    assert list(op(rows)) == etalon


def test_idf():
    row = {'word': 1, 'docs_with_words': 10, 'column_total_docs': 10}
    etalon = [{'word': 1, 'docs_with_words': 10, 'column_total_docs': 10, 'idf': 0.0}]
    op = operations.Idf('docs_with_words', 'column_total_docs')
    assert list(op(row)) == etalon

    row = {'word': 1, 'docs_with_words': 1, 'column_total_docs': 10}
    etalon = pytest.approx({'word': 1, 'docs_with_words': 1, 'column_total_docs': 10, 'idf': 2.302}, 0.001)
    assert next(op(row)) == etalon


def test_tf():
    rows = [{'doc': 1, 'word': 'hello'}, {'doc': 1, 'word': 'world'}, {'doc': 1, 'word': 'hello'},
            {'doc': 1, 'word': 'hello'}, {'doc': 1, 'word': 'world'}, {'doc': 1, 'word': 'hello'}]
    etalon = [pytest.approx({'doc': 1, 'word': 'hello', 'tf': 4/6}, 0.001), pytest.approx({'doc': 1, 'word': 'world', 'tf': 2/6}, 0.001)]
    op = operations.Tf('word')
    assert sorted(op(rows), key=lambda x: x['word']) == etalon


def test_product():
    row = {'operand_a': 1, 'operand_b': 2, 'operand_c': 3, 'operand_d': 4}
    etalon = {'operand_a': 1, 'operand_b': 2, 'operand_c': 3, 'operand_d': 4, 'product': 1}
    op = operations.Product('operand_a')
    assert next(op(row)) == etalon

    etalon = {'operand_a': 1, 'operand_b': 2, 'operand_c': 3, 'operand_d': 4, 'product': 2}
    op = operations.Product('operand_a', 'operand_b')
    assert next(op(row)) == etalon

    etalon = {'operand_a': 1, 'operand_b': 2, 'operand_c': 3, 'operand_d': 4, 'product': 8}
    op = operations.Product('operand_a', 'operand_b', 'operand_d')
    assert next(op(row)) == etalon


def test_divide():
    row = {'operand_a': 8, 'operand_b': 4}
    etalon = {'operand_a': 8, 'operand_b': 4, 'div': 2}
    op = operations.Divide('operand_a', 'operand_b')
    assert next(op(row)) == etalon

    row = {'operand_a': 4, 'operand_b': 8}
    etalon = {'operand_a': 4, 'operand_b': 8, 'div': 0.5}
    op = operations.Divide('operand_a', 'operand_b')
    assert next(op(row)) == etalon

    row = {'operand_a': 8, 'operand_b': 0}
    etalon = {'operand_a': 8, 'operand_b': 0, 'div': None}
    op = operations.Divide('operand_a', 'operand_b')
    assert next(op(row)) == etalon


def test_max():
    rows = [{'word': 'hello', 'count': 1}, {'word': 'hi', 'count': 10}, {'word': 'not', 'count': 2},
            {'word': 'lol', 'count': 5}, {'word': 'kek', 'count': 15}]
    etalon = [{'word': 'kek', 'count': 15}, {'word': 'hi', 'count': 10}]
    op = operations.Max('count', 2)
    assert sorted(op(rows), key=lambda x: x['count'], reverse=True) == etalon


def test_cut():
    row = {'a': 1, 'b': "a", 'c': '3'}
    etalon = [{'a': 1, 'c': "3"}]
    op = operations.Cut(['a', 'c'])
    assert list(op(row)) == etalon


def test_grep():
    row = {'a': 1, 'b': "artert", 'c': '3'}

    op = operations.Grep('a', lambda x: x > 0)
    assert list(op(row)) == [row]
    op = operations.Grep('a', lambda x: x < 0)
    assert list(op(row)) == []
    op = operations.Grep('b', lambda x: len(x) > 4)
    assert list(op(row)) == [row]
    op = operations.Grep('b', lambda x: len(x) < 4)
    assert list(op(row)) == []


def test_diff_time():
    row = {'enter_time': '20170912T123410.1794', 'leave_time': '20170912T123412.68'}
    etalon = [{'enter_time': '20170912T123410.1794', 'leave_time': '20170912T123412.68', 'diff_time': pytest.approx(2.5006, 0.001)}]

    op = operations.DiffTime('enter_time', 'leave_time')
    assert list(op(row)) == etalon


def test_week_hour_split_when_enter_and_leave_in_the_same_hour():
    row = {'enter_time': '20170912T123410.1794', 'leave_time': '20170912T123412.68'}
    etalon = {'weekday': 'Tue', 'enter_time': '20170912T123410.179400',
              'hour_time': pytest.approx(2.5006, 0.001), 'total_time': pytest.approx(2.5006, 0.001),
              'leave_time': '20170912T123412.680000', 'hour': 12}
    op = operations.WeekHourSplit('enter_time', 'leave_time')
    assert next(op(row)) == etalon


def test_week_hour_split_when_enter_and_leave_without_micros():
    row = {'enter_time': '20170912T123410', 'leave_time': '20170912T123412'}
    etalon = {'weekday': 'Tue', 'enter_time': '20170912T123410.000000',
              'hour_time': pytest.approx(2, 0.001), 'total_time': pytest.approx(2, 0.001),
              'leave_time': '20170912T123412.000000', 'hour': 12}
    op = operations.WeekHourSplit('enter_time', 'leave_time')
    assert next(op(row)) == etalon


def test_week_hour_split_when_enter_and_leave_in_different_hours():
    row = {'enter_time': '20170912T123410.1794', 'leave_time': '20170912T143912.68'}
    etalon = [pytest.approx({'leave_time': '20170912T130000.000000', 'hour_time': 1549.8206,
                             'enter_time': '20170912T123410.179400', 'hour': 12, 'total_time': 7502.5006, 'weekday': 'Tue'}, 0.001),
              pytest.approx({'leave_time': '20170912T140000.000000', 'hour_time': 3600.0,
                             'enter_time': '20170912T130000.000000', 'hour': 13, 'total_time': 7502.5006, 'weekday': 'Tue'}, 0.001),
              pytest.approx({'leave_time': '20170912T143912.680000', 'hour_time': 2352.68,
                             'enter_time': '20170912T140000.000000', 'hour': 14, 'total_time': 7502.5006, 'weekday': 'Tue'}, 0.001)]
    op = operations.WeekHourSplit('enter_time', 'leave_time')
    assert list(op(row)) == etalon


def test_week_hour_split_when_enter_and_leave_in_different_days():
    row = {'enter_time': '20170911T233410.1794', 'leave_time': '20170912T013412.68'}
    etalon = [pytest.approx({'hour_time': 1549.8206, 'leave_time': '20170912T000000.000000', 'enter_time': '20170911T233410.179400',
                             'total_time': 7202.5006, 'weekday': 'Mon', 'hour': 23}, 0.001),
              pytest.approx({'hour_time': 3600.0, 'leave_time': '20170912T010000.000000', 'enter_time': '20170912T000000.000000',
                             'total_time': 7202.5006, 'weekday': 'Tue', 'hour': 0}, 0.001),
              pytest.approx({'hour_time': 2052.68, 'leave_time': '20170912T013412.680000', 'enter_time': '20170912T010000.000000',
                             'total_time': 7202.5006, 'weekday': 'Tue', 'hour': 1}, 0.001)]
    op = operations.WeekHourSplit('enter_time', 'leave_time')
    assert list(op(row)) == etalon


def test_velocity():
    row = {'distance': 10, 'time': 1}
    etalon = {'distance': 10, 'time': 1, 'speed': 36}
    op = operations.Velocity('distance', 'time')
    assert next(op(row)) == etalon

    row = {'distance': 10, 'time': 0}
    etalon = {'distance': 10, 'time': 0, 'speed': None}
    op = operations.Velocity('distance', 'time')
    assert next(op(row)) == etalon


def test_inner_join():
    rows_a = [{'word': 'hello', 'count': 1}, {'word': 'hello', 'count': 10}]
    rows_b = [{'word': 'hello', 'other': 2}, {'word': 'hello', 'other': 12}]
    rows_empty = []

    etalon = [{'word': 'hello', 'count': 1, 'other': 2}, {'word': 'hello', 'count': 1, 'other': 12},
              {'word': 'hello', 'count': 10, 'other': 2}, {'word': 'hello', 'count': 10, 'other': 12}]

    op = operations.InnerJoiner()

    assert list(op(rows_a, rows_empty)) == []
    assert list(op(rows_empty, rows_a)) == []
    assert sorted(op(rows_a, rows_b), key=lambda x: (x['count'], x['other'])) == etalon


def test_outer_join():
    rows_a = [{'word': 'hello', 'count': 1}, {'word': 'hello', 'count': 10}]
    rows_b = [{'word': 'hello', 'other': 2}, {'word': 'hello', 'other': 12}]
    rows_empty = []

    etalon = [{'word': 'hello', 'count': 1, 'other': 2}, {'word': 'hello', 'count': 1, 'other': 12},
              {'word': 'hello', 'count': 10, 'other': 2}, {'word': 'hello', 'count': 10, 'other': 12}]

    op = operations.OuterJoiner()

    assert sorted(op(rows_a, rows_empty), key=lambda x: x['count']) == rows_a
    assert sorted(op(rows_empty, rows_a), key=lambda x: x['count']) == rows_a
    assert sorted(op(rows_a, rows_b), key=lambda x: (x['count'], x['other'])) == etalon


def test_left_join():
    rows_a = [{'word': 'hello', 'count': 1}, {'word': 'hello', 'count': 10}]
    rows_b = [{'word': 'hello', 'other': 2}, {'word': 'hello', 'other': 12}]
    rows_empty = []

    etalon = [{'word': 'hello', 'count': 1, 'other': 2}, {'word': 'hello', 'count': 1, 'other': 12},
              {'word': 'hello', 'count': 10, 'other': 2}, {'word': 'hello', 'count': 10, 'other': 12}]

    op = operations.LeftJoiner()

    assert sorted(op(rows_a, rows_empty), key=lambda x: x['count']) == rows_a
    assert sorted(op(rows_empty, rows_a), key=lambda x: x['count']) == []
    assert sorted(op(rows_a, rows_b), key=lambda x: (x['count'], x['other'])) == etalon


def test_right_join():
    rows_a = [{'word': 'hello', 'count': 1}, {'word': 'hello', 'count': 10}]
    rows_b = [{'word': 'hello', 'other': 2}, {'word': 'hello', 'other': 12}]
    rows_empty = []

    etalon = [{'word': 'hello', 'count': 1, 'other': 2}, {'word': 'hello', 'count': 1, 'other': 12},
              {'word': 'hello', 'count': 10, 'other': 2}, {'word': 'hello', 'count': 10, 'other': 12}]

    op = operations.RightJoiner()

    assert sorted(op(rows_a, rows_empty), key=lambda x: x['count']) == []
    assert sorted(op(rows_empty, rows_a), key=lambda x: x['count']) == rows_a
    assert sorted(op(rows_a, rows_b), key=lambda x: (x['count'], x['other'])) == etalon


def test_distance_from_lon_lat():
    row = {
        'start': [37.84870228730142, 55.73853974696249],
        'end': [37.8490418381989, 55.73832445777953]}
    etalon = {
        'start': [37.84870228730142, 55.73853974696249],
        'end': [37.8490418381989, 55.73832445777953],
        'length': pytest.approx(32.0238, 0.001)
    }
    op = operations.DistanceFromLonLat('start', 'end')

    assert next(op(row)) == etalon
