import string
import heapq
import math
from datetime import datetime, timedelta
from abc import abstractmethod
from copy import deepcopy
from math import sin, cos, sqrt, atan2, radians


class Mapper(object):
    """
    base class for mapping operations
    """
    @abstractmethod
    def __call__(self, r):
        """
        :param r: one table row
        :type r: dict[str,object]
        """
        pass


class Reducer(object):
    """
    base class for reduce operations
    """
    @abstractmethod
    def __call__(self, records):
        """
        :param records: table rows
        :type records: list[dict[str,object]]
        """
        pass


class Aggregator(object):
    """
    base class for folding operations
    """
    @abstractmethod
    def __call__(self, r, state):
        """
        :param r: one table row
        :type r: dict[str,object]
        :param state: mutable aggregation state
        :type state: dict[str,object]
        """
        pass


class Joiner(object):
    """
    base class for join operations
    """
    @abstractmethod
    def __call__(self, records_a, records_b):
        """
        :param records_a:
        :type records_a: list[dict[str, object]]
        :param records_b:
        :type records_b: list[dict[str, object]]
        :return: list[dict[str, object]]
        """
        pass


class FilterPunctuation(Mapper):
    """
    Left only non-punctuation symbols
    """
    def __init__(self, column):
        self.column = column

    @staticmethod
    def _filter_punctuation(txt):
        p = set(string.punctuation)
        return "".join([c for c in txt if c not in p])

    def __call__(self, r):
        new_r = deepcopy(r)
        new_r[self.column] = self._filter_punctuation(r[self.column])
        yield new_r


class LowerCase(Mapper):
    """
    Replace column value with value in lower case
    """
    def __init__(self, column):
        """
        :param column: name of column
        :type column: str
        """
        self.column = column

    @staticmethod
    def _lower_case(txt):
        return txt.lower()

    def __call__(self, r):
        new_r = deepcopy(r)
        new_r[self.column] = self._lower_case(r[self.column])
        yield new_r


class Grep(Mapper):
    """
    Remove records that don't satisfy some condition
    """
    def __init__(self, column, condition):
        """
        :param column: name of column
        :type column: str
        :param condition: if condition is not true - remove record
        :type condition: types.FunctionType
        """
        self.column = column
        self.condition = condition

    def __call__(self, r):
        if self.condition(r[self.column]):
            yield r


class Split(Mapper):
    """
    Split row on multiple rows by separator
    """
    def __init__(self, column, separator=None):
        """
        :param column: name of column
        :type column: str
        :param separator: string to separate by
        :type separator: str
        """
        self.column = column
        self.separator = separator

    @staticmethod
    def _split(txt, separator):
        for t in txt.split(separator):
            yield t

    def __call__(self, r):
        for t in self._split(r[self.column], self.separator):
            new_r = deepcopy(r)
            new_r[self.column] = t
            yield new_r


class Dummy(Mapper):
    """
    Pass row as is
    """
    def __call__(self, r):
        yield r


class Sum(Aggregator):
    """
    Add values passed to __call__ and save it in state
    """
    def __init__(self, column):
        """
        :param column: name of column
        :type column: str
        """
        self.column = column

    def __call__(self, r, state):
        if self.column not in state:
            state[self.column] = 0
        state[self.column] += r[self.column]
        return state


class Count(Aggregator):
    """
    Count values passed to __call__ and save it in state
    """
    def __init__(self, column):
        """
        :param column: name of column
        :type column: str
        """
        self.column = column

    def __call__(self, r, state):
        if self.column not in state:
            state[self.column] = 0
        state[self.column] += 1
        return state


class Velocity(Mapper):
    """
    Calculates velocity by time and distance
    """
    def __init__(self, column_distance, column_time, column_result='speed'):
        """
        :param column_distance: name for column with distance in meters
        :type column_distance: str
        :param column_time: name for column with time in seconds
        :type column_time: str
        :param column_result: name for result column with velocity in km/h
        :type column_result: str
        """
        self.column_distance = column_distance
        self.column_time = column_time
        self.column_result = column_result

    def __call__(self, r):
        new_r = deepcopy(r)
        new_r[self.column_result] = r[self.column_distance] / r[self.column_time] * 60 * 60 / 1000 if r[self.column_time] else None
        yield new_r


class FirstReducer(Reducer):
    """
    Yield only first record from records
    """
    def __call__(self, records):
        for r in records:
            yield r
            break


class DistanceFromLonLat(Mapper):
    """
    Calculates distance between geo-coordinates
    """

    R = 6373000

    def __init__(self, column_start, column_end, column_result='length'):
        """

        :param column_start: name for column with start point with lon, lat
        :type column_start: str
        :param column_end: name for column with start point with lon, lat
        :type column_end: str
        :param column_result: name for column with distance in meters
        :type column_result: float
        """
        self.column_start = column_start
        self.column_end = column_end
        self.column_result = column_result

    def __call__(self, r):

        lon1, lat1 = [radians(x) for x in r[self.column_start]]
        lon2, lat2 = [radians(x) for x in r[self.column_end]]

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        new_r = deepcopy(r)
        new_r[self.column_result] = self.R * c
        yield new_r


class Idf(Mapper):
    """
    Calculates log(column_total_docs) - log(column_docs_with_words)
    """
    def __init__(self, column_docs_with_words, column_total_docs, column_idf='idf'):
        """
        :type column_docs_with_words: str
        :type column_total_docs: str
        :type column_idf: str
        """
        self.column_docs_with_words = column_docs_with_words
        self.column_total_docs = column_total_docs
        self.column_idf = column_idf

    def __call__(self, r):
        new_r = deepcopy(r)
        new_r[self.column_idf] = math.log(r[self.column_total_docs]) - math.log(r[self.column_docs_with_words])
        yield new_r


class Tf(Reducer):
    """
    Calculate frequency of values in column
    """
    def __init__(self, column_words, column_tf='tf'):
        """
        :param column_words: name for column with words
        :type column_words: str
        :param column_tf: name for column with tf
        :type column_tf: str
        """
        self.column_words = column_words
        self.column_tf = column_tf

    def __call__(self, records):
        words_dict = {}
        last_record = {}
        for r in records:
            word = r[self.column_words]
            if word not in words_dict:
                words_dict[word] = 0
            words_dict[word] += 1
            last_record = r

        total = sum(words_dict.values())
        for k, v in words_dict.items():
            new_r = deepcopy(last_record)
            new_r[self.column_words] = k
            new_r[self.column_tf] = v/total
            yield new_r


class Product(Mapper):
    """
    Calculates product of multiple columns
    """
    def __init__(self, *columns_arguments, column_result='product'):
        """
        :param columns_arguments: column names to product
        :type columns_arguments: str
        :param column_result: column name to save product in
        :type column_result: str
        """
        self.columns_arguments = columns_arguments
        self.column_result = column_result

    def __call__(self, r):
        new_r = deepcopy(r)
        new_r[self.column_result] = 1
        for arg in self.columns_arguments:
            new_r[self.column_result] *= r[arg]
        yield new_r


class Max(Reducer):
    """
    Calculate top N by value
    """
    def __init__(self, column_max, n):
        """
        :param column_max: column names to find max
        :type column_max: str
        :param n: number of top values to extract
        :type n: int
        """
        self.column_max = column_max
        self.n = n

    def __call__(self, records):
        for r in heapq.nlargest(self.n, records, key=lambda x: x[self.column_max]):
            yield r


class DiffTime(Mapper):
    """
    Calculate difference between time columns
    """
    def __init__(self, column_a, column_b, column_result='diff_time'):
        """
        :param column_a: first time column name
        :type column_a: str
        :param column_b: second time column name
        :type column_b: str
        :param column_result: column name for difference in seconds
        :type column_result: str
        """
        self.column_a = column_a
        self.column_b = column_b
        self.column_result = column_result

    def __call__(self, r):
        da = datetime.strptime(r[self.column_a], "%Y%m%dT%H%M%S.%f")
        db = datetime.strptime(r[self.column_b], "%Y%m%dT%H%M%S.%f")

        new_r = deepcopy(r)
        new_r[self.column_result] = (db - da).total_seconds()
        yield new_r


class WeekHourSplit(Mapper):
    """
    Split yandex maps special log by week-hour and set time interval on each of them
    """
    def __init__(self, column_start, column_end, hour_column='hour', week_column='weekday',
                 hour_time_column='hour_time', total_time_column="total_time"):
        """

        :param column_start: name of column with start time in format self.dt_template_*. Replaced by interval start
        :type column_start: str
        :param column_end: name of column with end time in format self.dt_template_*. Replaced by interval end
        :type column_end: str
        :param hour_column: name of column to save integer hour
        :type hour_column: str
        :param week_column: name of column to save string weekday
        :type week_column: str
        :param hour_time_column: name of column to save time spend on this hour
        :type hour_time_column: str
        :param total_time_column: name of column to save time spend in total
        :type total_time_column: str
        """
        self.column_start = column_start
        self.column_end = column_end
        self.hour_column = hour_column
        self.week_column = week_column
        self.hour_time_column = hour_time_column
        self.total_time_column = total_time_column
        self.dt_template = "%Y%m%dT%H%M%S.%f"
        self.dt_template_backup = "%Y%m%dT%H%M%S"

    def __call__(self, r):

        if '.' in r[self.column_start]:
            da = datetime.strptime(r[self.column_start], self.dt_template)
        else:
            da = datetime.strptime(r[self.column_start], self.dt_template_backup)

        if '.' in r[self.column_end]:
            db = datetime.strptime(r[self.column_end], self.dt_template)
        else:
            db = datetime.strptime(r[self.column_end], self.dt_template_backup)

        total_time = (db - da).total_seconds()
        step_da = da
        step_db = min((step_da + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0), db)
        while step_da < db:
            new_r = deepcopy(r)
            new_r[self.hour_column] = step_da.hour
            new_r[self.week_column] = step_da.strftime('%a')
            new_r[self.hour_time_column] = (step_db - step_da).total_seconds()
            new_r[self.total_time_column] = total_time

            new_r[self.column_start] = step_da.strftime(self.dt_template)
            new_r[self.column_end] = step_db.strftime(self.dt_template)

            yield new_r
            step_da = step_db
            step_db = min((step_da + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0), db)


class Divide(Mapper):
    """
    Mapper for dividing two values. If division cannot be provided - save None in result.
    """
    def __init__(self, column_numerator, column_denominator, column_result='div'):
        """
        :type column_numerator: str
        :type column_denominator: str
        :type column_result: str
        """
        self.column_numerator = column_numerator
        self.column_denominator = column_denominator
        self.column_result = column_result

    def __call__(self, r):
        new_r = deepcopy(r)
        new_r[self.column_result] = r[self.column_numerator] / r[self.column_denominator] if r[self.column_denominator] else None
        yield new_r


class InnerJoiner(Joiner):
    """
    Join with inner strategy
    """
    @abstractmethod
    def __call__(self, records_a, records_b):
        cache = list(records_b)
        for a in records_a:
            for b in cache:
                new_r = deepcopy(a)
                new_r.update(b)
                yield new_r


class OuterJoiner(Joiner):
    """
    Join with outer strategy
    """
    @abstractmethod
    def __call__(self, records_a, records_b):
        cache_a = list(records_a)
        cache_b = list(records_b)

        if not cache_a:
            for b in cache_b:
                yield b

        if not cache_b:
            for a in cache_a:
                yield a

        for a in cache_a:
            for b in cache_b:
                new_r = deepcopy(a)
                new_r.update(b)
                yield new_r


class LeftJoiner(Joiner):
    """
    Join with left strategy
    """
    @abstractmethod
    def __call__(self, records_a, records_b):
        cache_b = list(records_b)

        for a in records_a:
            for b in cache_b:
                new_r = deepcopy(a)
                new_r.update(b)
                yield new_r
            if not cache_b:
                yield a


class RightJoiner(Joiner):
    """
    Join with right strategy
    """
    @abstractmethod
    def __call__(self, records_a, records_b):
        cache_a = list(records_a)

        for b in records_b:
            for a in cache_a:
                new_r = deepcopy(b)
                new_r.update(a)
                yield new_r
            if not cache_a:
                yield b


class Cut(Mapper):
    """
    Leave only mentioned columns
    """
    def __init__(self, columns):
        """
        :type columns: list[str]
        """
        self.columns = columns

    def __call__(self, r):
        yield {k: r[k] for k in self.columns}
