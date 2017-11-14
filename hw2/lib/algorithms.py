from hw2.lib import mr
from hw2.lib import operations


def build_word_count_graph(input_stream, text_column, count_column):
    return mr.FireMR()\
        .read_from_iter(input_stream)\
        .map(operations.FilterPunctuation(text_column))\
        .map(operations.LowerCase(text_column))\
        .map(operations.Split(text_column))\
        .sort([text_column])\
        .aggregate(operations.Count(count_column), [text_column])\
        .sort([count_column, text_column])


def word_count(input_stream, text_column, count_column):
    """
    Task 1
    """
    buffer = []
    g = build_word_count_graph(input_stream, text_column, count_column)
    g = g.save(buffer)
    g.run()
    return buffer


def build_inverted_index_graph(input_stream, doc_column, text_column):
    read_stream = mr.FireMR() \
        .read_from_iter(input_stream)

    split_by_words = read_stream \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    count_docs = read_stream \
        .aggregate(operations.Count('total_doc_count'), [])

    idf = split_by_words.sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .aggregate(operations.Count('doc_with_word_count'), [text_column]) \
        .join(operations.InnerJoiner(), count_docs, []) \
        .map(operations.Idf('doc_with_word_count', 'total_doc_count')) \
        .sort([text_column])

    tf = split_by_words.sort([doc_column]) \
        .reduce(operations.Tf(text_column), [doc_column]) \
        .sort([text_column])

    tf_idf_result = tf.join(operations.InnerJoiner(), idf, [text_column]) \
        .map(operations.Product('tf', 'idf', column_result='tf_idf'))

    top3 = tf_idf_result.sort([text_column]) \
        .reduce(operations.Max('tf_idf', 3), [text_column]) \
        .map(operations.Cut([text_column, doc_column, 'tf_idf']))

    return top3


def tf_idf(input_stream, doc_column, text_column):
    """
    Task 2
    """
    buffer = []
    g = build_inverted_index_graph(input_stream, doc_column, text_column)
    g = g.save(buffer)
    g.run()
    return buffer


def build_pmi_graph(input_stream, doc_column, text_column):
    read_stream = mr.FireMR() \
        .read_from_iter(input_stream)

    split_by_words = read_stream \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([doc_column, text_column])

    count_words_in_doc = split_by_words.sort([doc_column, text_column]) \
        .aggregate(operations.Count('count'), [doc_column, text_column])

    filtered_words = split_by_words.join(operations.OuterJoiner(), count_words_in_doc, [doc_column, text_column]) \
        .map(operations.Grep('count', lambda x: x >= 2)) \
        .map(operations.Grep(text_column, lambda x: len(x) >= 4))

    tf = filtered_words.sort([doc_column]) \
        .reduce(operations.Tf(text_column), [doc_column]) \
        .sort([text_column])

    total_tf = filtered_words.reduce(operations.Tf(text_column, 'total_tf'), [])\
        .map(operations.Cut(['total_tf', text_column]))\
        .sort([text_column])

    result = tf.join(operations.OuterJoiner(), total_tf, [text_column]) \
        .map(operations.Idf('total_tf', 'tf', 'pmi'))\
        .sort([doc_column]) \
        .reduce(operations.Max('pmi', 10), [doc_column]) \
        .map(operations.Cut([text_column, doc_column, 'pmi']))

    return result


def pmi(input_stream, doc_column, text_column):
    """
    Task 3
    """
    buffer = []
    g = build_pmi_graph(input_stream, doc_column, text_column)
    g = g.save(buffer)
    g.run()
    return buffer


def build_yandex_maps_graph(input_stream_time, input_stream_length):
    read_time_stream = mr.FireMR() \
        .read_from_iter(input_stream_time) \
        .map(operations.WeekHourSplit('enter_time', 'leave_time')) \
        .sort(['edge_id'])

    read_length_stream = mr.FireMR() \
        .read_from_iter(input_stream_length) \
        .map(operations.DistanceFromLonLat('start', 'end')) \
        .sort(['edge_id'])

    agg = read_time_stream \
        .join(operations.InnerJoiner(), read_length_stream, ['edge_id']) \
        .map(operations.Product('length', 'hour_time', column_result='hour_length')) \
        .map(operations.Divide('hour_length', 'total_time', column_result='hour_length')) \
        .map(operations.Cut(['weekday', 'hour', 'hour_length', 'hour_time'])) \
        .sort(['weekday', 'hour'])

    length_agg = agg \
        .aggregate(operations.Sum('hour_length'), keys=['weekday', 'hour']) \
        .sort(['weekday', 'hour'])

    time_agg = agg \
        .aggregate(operations.Sum('hour_time'), keys=['weekday', 'hour']) \
        .sort(['weekday', 'hour'])

    result = length_agg \
        .join(operations.InnerJoiner(), time_agg, ['weekday', 'hour']) \
        .map(operations.Velocity('hour_length', 'hour_time')) \
        .map(operations.Cut(['weekday', 'hour', 'speed']))

    return result


def yandex_maps(input_stream_time, input_stream_length, verbose=True):
    """
    Task 4
    """
    buffer = []
    g = build_yandex_maps_graph(input_stream_time, input_stream_length)
    g = g.save(buffer)
    g.run(verbose=verbose)
    return buffer
