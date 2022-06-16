import math
from pprint import pprint
import pandas as pd
import numpy as np


# r - correlation, n - samples
# se - standard error
# cez - common effect size
# rew - random effect weight
from settings import STATS_DIR, STATS_RESULTS_DIR


def common_effect_size_vector(r_vector, debug=True):
    f = lambda x: x if (x != 1 and x != -1) else 0.99 * x
    r_vector = np.array(list(map(f, r_vector)))
    if debug:
        print(f'r vector: {r_vector}')
    return np.arctanh(r_vector)


def estimated_standard_error_vector(n_vector):
    f = lambda x: 1 / (x - 3) if x > 3 else 1
    return np.array(list(map(f, n_vector)))
    # return f(n_vector)


def estimated_weight_vector(n_vector):
    f = lambda x: x - 3 if x > 3 else 0
    return np.array(list(map(f, n_vector)))


def c_statistic(weight_vector):
    weight_sum = np.sum(weight_vector)
    weight_squared_sum = np.sum(np.square(weight_vector))
    return weight_sum - (weight_squared_sum / weight_sum) if weight_sum != 0 else 0


def cochrans_q_statistic(weight_vector, cez_vector):
    weight_sum = np.sum(weight_vector)
    # print(weight_vector)
    weighted_cez_sum_squared = np.square(np.sum(weight_vector * cez_vector))
    weighted_cez_squared_sum = np.sum(weight_vector * np.square(cez_vector))
    # print(weight_sum, weighted_cez_squared_sum, weighted_cez_sum_squared)
    return weighted_cez_squared_sum - (weighted_cez_sum_squared / weight_sum) if weight_sum != 0 else 0


def estimated_between_study_variance(q, c, df):
    return (q - df) / c if c != 0 else 1


# bsv - that^2
def estimated_random_effect_weight(se_vector, bsv):
    plus_bsv = lambda x: x + bsv
    reverse = lambda x: 1 / x
    return reverse(plus_bsv(se_vector))


def estimated_random_effect_cez(rew_vector, cez_vector):
    weight_sum = np.sum(rew_vector)
    weighted_cez_sum = np.sum(rew_vector * cez_vector)
    return weighted_cez_sum / weight_sum if weight_sum != 0 else 1


def standard_error_of_estimator(rew_vector):
    weight_sum = np.sum(rew_vector)
    # print(weight_sum)
    return math.sqrt(1 / weight_sum) if weight_sum > 0 else 1


def lower_limit_confidence_interval(cez_re, se_re):
    transformed = cez_re - 1.96 * se_re
    not_transformed = math.tanh(transformed)
    return [transformed, not_transformed]


def upper_limit_confidence_interval(cez_re, se_re):
    transformed = cez_re + 1.96 * se_re
    not_transformed = math.tanh(transformed)
    return [transformed, not_transformed]


def get_statistical_values(sample_sizes, correlations, number_of_projects, debug=True):
    if len(correlations) == 0:
        NaN = np.nan
        return [NaN, NaN, NaN, [NaN, NaN], [NaN, NaN]]
    [r, n, k] = [correlations, sample_sizes, number_of_projects]
    theta = common_effect_size_vector(r, debug=False)
    v = estimated_standard_error_vector(n)
    w = estimated_weight_vector(n)
    q = cochrans_q_statistic(w, theta)
    c = c_statistic(w)
    t_hat_squared = estimated_between_study_variance(q, c, k - 1)
    w_prime = estimated_random_effect_weight(v, t_hat_squared)

    theta_re = estimated_random_effect_cez(w_prime, theta)
    p = math.tanh(theta_re)
    se_re = standard_error_of_estimator(w_prime)
    LL = lower_limit_confidence_interval(theta_re, se_re)
    UL = upper_limit_confidence_interval(theta_re, se_re)
    if debug:
        print(
            f"theta:{theta}\n"  # v
            f"v:{v}\n"  # v variance
            f"w:{w}\n"  # v
            f"q:{q}\n"  # v
            f"c:{c}\n"  # v
            f"t_hat:{t_hat_squared}\n"  # v
            f"w_prime:{w_prime}\n"  # v
            f"theta_re:{theta_re}\n"  # v
            f"p:{p}\n"
            f"standard_error_of_estimator:{se_re}\n"  # v
            f"LL:{LL}\n"
            f"UL:{UL}\n"
        )
    return [theta_re, p, se_re, LL, UL]


def get_pure_correlation_dataframe(samples, correlations):
    import pandas as pd
    data = pd.concat([samples, correlations], axis=1)
    data = data.dropna(axis=0)
    return data


# TODO: refactor
def get_statistical_values_from_df(df):
    import pandas as pd
    k = len(df.index)
    samples = df.loc[:, 'n']
    thetav = []
    pv = []
    se_rev = []
    LLv = []
    ULv = []
    for col in df.columns:
        if col != 'n':
            correlations = df.loc[:, col]
            pure_data = get_pure_correlation_dataframe(samples, correlations)
            pure_samples = np.array(pure_data.iloc[:, 0])
            pure_correlations = np.array(pure_data.iloc[:, 1])
            stats = get_statistical_values(pure_samples, pure_correlations, k, debug=False)
            [theta_re, p, se_re, LL, UL] = stats
            thetav.append(theta_re)
            pv.append(p)
            se_rev.append(se_re)
            LLv.append(LL[1])
            ULv.append(UL[1])
    add_data = pd.DataFrame([thetav, pv, se_rev, LLv, ULv])
    index = df.columns.values[1:]
    add_data = add_data.rename(lambda x: index[x], axis='columns')
    add_data = add_data.rename(lambda x: ['theta', 'p', 'se', 'll', 'ul'][x], axis='rows')
    return add_data
    # print(add_data)


def add_stats_to_correlation_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    add_stats = get_statistical_values_from_df(df)
    f = pd.concat([df, add_stats])
    f.to_csv(path)


def analyze_language_results(lang):
    from pathlib import Path
    dir_path = f"./{STATS_DIR}/{STATS_RESULTS_DIR}/{lang}"
    for csv_path in Path(dir_path).rglob('*.csv'):
        add_stats_to_correlation_csv(csv_path)


def get_metrics_name(index, metrics_amount):
    get_correlated_name = lambda index_val: index_val.split(' x ')[1]
    correlated_names = list(map(get_correlated_name, index))
    return [index[0].split(' x ')[0]] + correlated_names


def matrix_from_series(series) -> pd.DataFrame:
    series = series.iloc[1:]  # remove first entry that is sample size
    index = series.index
    index_len = len(index)
    series.reset_index(drop=True, inplace=True)
    metrics_amount = round(1 / 2 * (1 + (1 + 8 * index_len) ** (1 / 2)))
    df = pd.DataFrame()
    for i in range(metrics_amount - 1):
        x, inc_i = metrics_amount, i + 1
        interval = list(map(lambda k: int(k * (x - (k + 1) / 2)), [i, inc_i]))
        series_as_row = series.iloc[interval[0]:interval[1]].to_frame().T
        series_as_row = series_as_row.rename(lambda a: a - interval[0] + i, axis='columns')
        df = pd.concat([df, series_as_row], axis=0, ignore_index=True)
    empty = pd.DataFrame(np.nan, index=range(metrics_amount), columns=['A'])
    df = pd.concat([empty, df], axis=1, ignore_index=True)
    metric_names = get_metrics_name(index, metrics_amount)
    df = df.rename(lambda a: metric_names[a])
    df = df.rename(lambda a: metric_names[a], axis=1)
    return df


def result_correlation_matrix_from_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return matrix_from_series(df.loc['p'])


def result_se_matrix_from_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return matrix_from_series(df.loc['se'])


def generate_stats_matrices_from_csv(path):
    path = str(path)
    corr: pd.DataFrame = result_correlation_matrix_from_csv(path)
    se: pd.DataFrame = result_se_matrix_from_csv(path)
    corr_filename = path.split('.csv')[0] + '-Correlation.csv'
    se_filename = path.split('.csv')[0] + '-StandardError.csv'
    corr.to_csv(corr_filename)
    se.to_csv(se_filename)


def generate_stats_matrices_in_dir(dir_path: str):
    from pathlib import Path
    for csv_path in Path(dir_path).rglob('*.csv'):
        generate_stats_matrices_from_csv(csv_path)


def generate_stats_matrices(lang: str):
    generate_stats_matrices_in_dir(f"./{STATS_DIR}/{STATS_RESULTS_DIR}/{lang}/")


if __name__ == '__main__':
    ex_path = './stats/results/JS_results/Attribute2.csv'
    # generate_stats_matrices()
    # filter_filled_columns_from_file(ex_path)
    # add_stats_to_correlation_csv(ex_path)
    # remove_empty_columns()
    get_statistical_values(sample_sizes=[100, 200, 300], correlations=[-1, 1, 0.1], number_of_projects=3)
