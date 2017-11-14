import json
from hw2.lib.algorithms import yandex_maps

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def check_yandex_map():
    with open("../resource/graph_data.txt") as length:
        with open("../resource/travel_times.txt") as time:
            length_iter = (json.loads(s) for s in length)
            time_iter = (json.loads(s) for s in time)
            result = yandex_maps(time_iter, length_iter, verbose=True)

    hours = list(range(24))
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    data = [[0] * 24 for _ in range(7)]

    with open('../resource/trafic.txt', 'w') as f:
        for r in result:
            f.write(json.dumps(r))
            f.write('\n')
            data[weekdays.index(r['weekday'])][hours.index(r['hour'])] = r['speed']

    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.savefig('../resource/trafic.png')


if __name__ == "__main__":
    check_yandex_map()
