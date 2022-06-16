import pandas as pd

from settings import STATS_RESULTS_DIR, STATS_DIR


def create_folder(path):
    import os
    # Check whether the specified path exists or not
    is_exist = os.path.exists(path)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path)


def remove_empty_vectors_file(path):
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    return df


# Remove empty columns and rows in statistic result files of specified language
def remove_empty_vectors(lang: str):
    from pathlib import Path
    dir_path = f"{STATS_DIR}/{STATS_RESULTS_DIR}/{lang}"
    for csv_path in Path(dir_path).rglob('*.csv'):
        df = remove_empty_vectors_file(csv_path)
        df.to_csv(csv_path)
