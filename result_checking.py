import math
from pathlib import Path

import numpy
import pandas
import pandas as pd
from matplotlib import pyplot as plt


def get_column_values_from_file(path, column) -> pandas.Series:
    df = pd.read_csv(path)
    return df.loc[:, column]


def get_all_metric_values(metric, level, lang='JavaScript'):
    dir_path = f'./results/{lang}/'
    res = []
    for csv_path in Path(dir_path).rglob(f'*-{level}.csv'):
        # print("sas")
        res.extend(get_column_values_from_file(csv_path, metric).tolist())
    return res


if __name__ == '__main__':
    # c = get_column_values_from_file(
    #     './results/JavaScript/chartjs__Chart.js/chartjs__Chart.js/' +
    #     'javascript/2022-04-11-01-12-13/chartjs__Chart.js-Function.csv', column='McCC')
    c = numpy.array(get_all_metric_values(metric="LOC", level="Function"))
    # print(c)
    c = numpy.array(list(map(lambda x: x if x > 1 else 0, c)))
    plt.yscale('log')
    plt.hist(c, bins=100)
    plt.show()
