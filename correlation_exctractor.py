from pathlib import Path

from settings import STATS_DIR, STATS_RESULTS_DIR
from utils import create_folder


def generate_correlation_matrix_from_csv(path, debug=True, dropna=False):
    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns
    import numpy as np
    f = pd.read_csv(path)
    corr = f.corr(method='spearman')

    if dropna:
        corr = corr.dropna(axis=1, how='all').dropna(axis=0, how='all')

    corr = pd.DataFrame(np.triu(corr), index=corr.columns, columns=corr.columns)
    if debug:
        if len(corr.index) > 0:
            plt.figure(figsize=(40, 20))
            sns.heatmap(corr, cmap="Greens")
            plt.show()
        else:
            print("Empty table")

    return corr
    # corr.to_csv("example.csv")


def generate_index(labels):
    length = len(labels)
    result = []
    for i in range(length):
        for j in range(i + 1, length):
            result.append(f"{labels[i]} x {labels[j]}")
    return result


def generate_correlation_row_from_csv(path, debug=True):
    import pandas as pd
    f = pd.read_csv(path)
    n = len(f.index)

    corr = generate_correlation_matrix_from_csv(path, debug=False)

    corr_len = len(corr.index)
    if corr_len == 0:
        return -1
    else:
        # Squeeze upper triangle in one row
        row = corr.iloc[0, 1:]
        for i in range(1, corr_len - 1):
            row = pd.concat([row, corr.iloc[i, i + 1:]], ignore_index=True)
        # Assign labels to each value in following format: metric_a x metric_b
        index = generate_index(corr.columns.values)
        row = row.rename(lambda x: index[x])
        # Add sample size to the row
        n_data = pd.Series(data={'n': n}, index=['n'])
        row = pd.concat([n_data, row])
        # Transform to dataframe row
        row = row.to_frame().transpose()
        return row


def concat_correlation_rows(path1, path2, debug=True):
    import pandas as pd
    f1 = pd.read_csv(path1)
    f2 = pd.read_csv(path2)
    f = pd.concat([f1, f2])
    if debug:
        print(f1)
        print(f2)
        print(f)
        f.to_csv('stats/concat.csv')
    return f


def get_file_type(path):
    import os
    filename = os.path.basename(path)
    name = filename.split('.csv')[0]
    return name.split('-')[-1]


def concat_project_stats_to_results(project_result_path, lang):
    import shutil
    for csv_path in Path(project_result_path).rglob('*.csv'):
        file_type = get_file_type(csv_path)
        dist = f"./{STATS_DIR}/{STATS_RESULTS_DIR}/{lang}/{file_type}.csv"
        try:
            f = concat_correlation_rows(dist, csv_path, debug=False)
            f.to_csv(dist, index=False)
        except:
            create_folder(f"./{STATS_DIR}/{STATS_RESULTS_DIR}/{lang}")
            shutil.copy(csv_path, dist)


def export_data_from_json(path, debug=True):
    import json
    import pandas as pd
    import pprint as pp
    with open(path) as f:
        json_data = json.loads(f.read())
        nodes = json_data['nodes']
        nodes_metrics = None
        for node in nodes:
            attributes = node['attributes']
            metrics = {}
            for attribute in attributes:
                if attribute['context'] == 'metric':
                    metrics[attribute['name']] = attribute['value']
            if debug:
                pp.pprint(pd.DataFrame([metrics]))
            if nodes_metrics is None:
                nodes_metrics = pd.DataFrame([metrics])
            else:
                nodes_metrics = nodes_metrics.append(pd.DataFrame([metrics]))
    return nodes_metrics


def analyze_project(path, project_name, lang, debug=False):
    from pathlib import Path
    from utils import create_folder
    create_folder(f'{STATS_DIR}/{lang}/{project_name}')
    for csv_path in Path(path).rglob('*.csv'):
        if debug:
            print(csv_path.name)
        corr = generate_correlation_row_from_csv(csv_path, debug=False)
        if type(corr).__name__ == 'int':  # empty matrix
            continue
        corr.to_csv(f'{STATS_DIR}/{lang}/{project_name}/{csv_path.name}', index=False)


if __name__ == '__main__':
    print("Testing functionality")
    # generate_correlation_row_from_csv('Java/Results/myproj/java/2021-11-18-03-34-33/myproj-Interface.csv')
    # export_data_from_json('Java/Results/myproj/java/2021-11-18-03-34-33/myproj-summary.json')
    # analyze_project(
    #     'results/JavaScript/cleanlock__VideoAdBlockForTwitch/cleanlock__VideoAdBlockForTwitch/javascript/2022-04-03-23-40-51')
    # concat_correlation_rows('./stats/brave__brave-browser/brave__brave-browser-Attribute.csv',
    #                         './stats/cleanlock__VideoAdBlockForTwitch/cleanlock__VideoAdBlockForTwitch-Attribute.csv')
    # print(get_file_type('stats/example2js/brave__brave-browser-Attribute.csv'))
    # concat_project_stats_to_results("stats/brave__brave-browser")
