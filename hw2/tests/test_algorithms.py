import pytest
from hw2.lib import algorithms


def test_word_count():
    rows = [{'doc_id': 1, 'text': 'hello, my little WORLD'}, {'doc_id': 2, 'text': 'Hello, my little little hell'}]
    etalon = [
        {'count': 1, 'text': 'hell'},
        {'count': 1, 'text': 'world'},
        {'count': 2, 'text': 'hello'},
        {'count': 2, 'text': 'my'},
        {'count': 3, 'text': 'little'}
    ]
    result = algorithms.word_count(rows, 'text', 'count')
    assert result == etalon


def test_tf_idf():
    rows = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    etalon = [
        pytest.approx({"text": "hello",  "doc_id": 5, "tf_idf": 0.2703}, 0.001),
        pytest.approx({"text": "hello", "doc_id": 1, "tf_idf": 0.1351}, 0.001),
        pytest.approx({"text": "hello", "doc_id": 4, "tf_idf": 0.1013}, 0.001),
        pytest.approx({"text": "little", "doc_id": 2, "tf_idf": 0.4054}, 0.001),
        pytest.approx({"text": "little", "doc_id": 3, "tf_idf": 0.4054}, 0.001),
        pytest.approx({"text": "little", "doc_id": 4, "tf_idf": 0.2027}, 0.001),
        pytest.approx({"text": "world", "doc_id": 6, "tf_idf": 0.3243}, 0.001),
        pytest.approx({"text": "world", "doc_id": 1, "tf_idf": 0.1351}, 0.001),
        pytest.approx({"text": "world", "doc_id": 5, "tf_idf": 0.1351}, 0.001)
    ]

    result = algorithms.tf_idf(rows, 'doc_id', 'text')
    assert result == etalon


def test_pmi():
    rows = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!'}
    ]

    etalon = [
        pytest.approx({"doc_id": 3, "text": "little", "pmi": 0.9555}, 0.001),
        pytest.approx({"doc_id": 4, "text": "little", "pmi": 0.9555}, 0.001),
        pytest.approx({"doc_id": 5, "text": "hello", "pmi": 1.1786}, 0.001),
        pytest.approx({"doc_id": 6, "text": "world", "pmi": 0.7731}, 0.001),
        pytest.approx({"doc_id": 6, "text": "hello", "pmi": 0.0800}, 0.001),
    ]

    result = algorithms.pmi(rows, 'doc_id', 'text')
    assert result == etalon


def test_yandex_maps():
    lengths = [
        {"start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
         "edge_id": 8414926848168493057},
        {"start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
         "edge_id": 5342768494149337085},
        {"start": [37.56963176652789, 55.846845586784184], "end": [37.57018438540399, 55.8469259692356],
         "edge_id": 5123042926973124604},
        {"start": [37.41463478654623, 55.654487907886505], "end": [37.41442892700434, 55.654839486815035],
         "edge_id": 5726148664276615162},
        {"start": [37.584684155881405, 55.78285809606314], "end": [37.58415022864938, 55.78177368734032],
         "edge_id": 451916977441439743},
        {"start": [37.736429711803794, 55.62696328852326], "end": [37.736344216391444, 55.626937723718584],
         "edge_id": 7639557040160407543},
        {"start": [37.83196756616235, 55.76662947423756], "end": [37.83191015012562, 55.766647034324706],
         "edge_id": 1293255682152955894},
    ]

    times = [
        {"leave_time": "20171020T112238.723000", "enter_time": "20171020T112237.427000", "edge_id": 8414926848168493057},
        {"leave_time": "20171011T145553.040000", "enter_time": "20171011T145551.957000", "edge_id": 8414926848168493057},
        {"leave_time": "20171020T090548.939000", "enter_time": "20171020T090547.463000", "edge_id": 8414926848168493057},
        {"leave_time": "20171024T144101.879000", "enter_time": "20171024T144059.102000", "edge_id": 8414926848168493057},
        {"leave_time": "20171022T131828.330000", "enter_time": "20171022T131820.842000", "edge_id": 5342768494149337085},
        {"leave_time": "20171014T134826.836000", "enter_time": "20171014T134825.215000", "edge_id": 5342768494149337085},
        {"leave_time": "20171010T060609.897000", "enter_time": "20171010T060608.344000", "edge_id": 5342768494149337085},
        {"leave_time": "20171027T082600.201000", "enter_time": "20171027T082557.571000", "edge_id": 5342768494149337085}
    ]

    etalon = [
        {'hour': 8, 'speed': pytest.approx(62.2322, 0.001), 'weekday': 'Fri'},
        {'hour': 9, 'speed': pytest.approx(78.1070, 0.001), 'weekday': 'Fri'},
        {'hour': 11, 'speed': pytest.approx(88.9552, 0.001), 'weekday': 'Fri'},
        {'hour': 13, 'speed': pytest.approx(100.9690, 0.001), 'weekday': 'Sat'},
        {'hour': 13, 'speed': pytest.approx(21.8577, 0.001), 'weekday': 'Sun'},
        {'hour': 6, 'speed': pytest.approx(105.3901, 0.001), 'weekday': 'Tue'},
        {'hour': 14, 'speed': pytest.approx(41.5145, 0.001), 'weekday': 'Tue'},
        {'hour': 14, 'speed': pytest.approx(106.4505, 0.001), 'weekday': 'Wed'}
    ]
    result = algorithms.yandex_maps(times, lengths)
    assert sorted(result, key=lambda x: (x['weekday'], x['hour'])) == sorted(etalon, key=lambda x: (x['weekday'], x['hour']))


def test_yandex_maps_iter():
    lengths = (i for i in [
        {"start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
         "edge_id": 8414926848168493057},
        {"start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
         "edge_id": 5342768494149337085},
        {"start": [37.56963176652789, 55.846845586784184], "end": [37.57018438540399, 55.8469259692356],
         "edge_id": 5123042926973124604},
        {"start": [37.41463478654623, 55.654487907886505], "end": [37.41442892700434, 55.654839486815035],
         "edge_id": 5726148664276615162},
        {"start": [37.584684155881405, 55.78285809606314], "end": [37.58415022864938, 55.78177368734032],
         "edge_id": 451916977441439743},
        {"start": [37.736429711803794, 55.62696328852326], "end": [37.736344216391444, 55.626937723718584],
         "edge_id": 7639557040160407543},
        {"start": [37.83196756616235, 55.76662947423756], "end": [37.83191015012562, 55.766647034324706],
         "edge_id": 1293255682152955894},
    ])

    times = (i for i in [
        {"leave_time": "20171020T112238.723000", "enter_time": "20171020T112237.427000", "edge_id": 8414926848168493057},
        {"leave_time": "20171011T145553.040000", "enter_time": "20171011T145551.957000", "edge_id": 8414926848168493057},
        {"leave_time": "20171020T090548.939000", "enter_time": "20171020T090547.463000", "edge_id": 8414926848168493057},
        {"leave_time": "20171024T144101.879000", "enter_time": "20171024T144059.102000", "edge_id": 8414926848168493057},
        {"leave_time": "20171022T131828.330000", "enter_time": "20171022T131820.842000", "edge_id": 5342768494149337085},
        {"leave_time": "20171014T134826.836000", "enter_time": "20171014T134825.215000", "edge_id": 5342768494149337085},
        {"leave_time": "20171010T060609.897000", "enter_time": "20171010T060608.344000", "edge_id": 5342768494149337085},
        {"leave_time": "20171027T082600.201000", "enter_time": "20171027T082557.571000", "edge_id": 5342768494149337085}
    ])

    etalon = [
        {'hour': 8, 'speed': pytest.approx(62.2322, 0.001), 'weekday': 'Fri'},
        {'hour': 9, 'speed': pytest.approx(78.1070, 0.001), 'weekday': 'Fri'},
        {'hour': 11, 'speed': pytest.approx(88.9552, 0.001), 'weekday': 'Fri'},
        {'hour': 13, 'speed': pytest.approx(100.9690, 0.001), 'weekday': 'Sat'},
        {'hour': 13, 'speed': pytest.approx(21.8577, 0.001), 'weekday': 'Sun'},
        {'hour': 6, 'speed': pytest.approx(105.3901, 0.001), 'weekday': 'Tue'},
        {'hour': 14, 'speed': pytest.approx(41.5145, 0.001), 'weekday': 'Tue'},
        {'hour': 14, 'speed': pytest.approx(106.4505, 0.001), 'weekday': 'Wed'}
    ]
    result = algorithms.yandex_maps(times, lengths)
    assert sorted(result, key=lambda x: (x['weekday'], x['hour'])) == sorted(etalon, key=lambda x: (x['weekday'], x['hour']))