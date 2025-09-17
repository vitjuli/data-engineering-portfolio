"""
Common reusable functions and classes extracted from notebooks.
Author: Iuliia Vitiugova
"""

from scipy.stats import shapiro, levene

# Function to check normality
def check_normality(data, group, period):
    stat, p = shapiro(data)
    print(f"Group {group}, Period {period}: W-statistic={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print("    Data is normally distributed.")
    else:
        print("    Data is not normally distributed.")
    return p > 0.05

# Checking normality for each group
for period, data in zip(["14-28 days", "28-42 days"], [period_14_28, period_28_42]):
    print(f"\nPeriod: {period}")
    for group in ['D1', 'D2', 'D3']:
        group_data = data[group].dropna()
        check_normality(group_data, group, period)

# Function to check equality of variances using Levene's test
def check_variance_equality(data1, data2, data3, period):
    stat, p = levene(data1, data2, data3)
    print(f"\nEquality of variances for period {period}:")
    print(f"Statistic={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print("    Variances are equal.")
    else:
        print("    Variances are not equal.")

# Applying Levene's test for each period
check_variance_equality(
    period_14_28['D1'].dropna(),
    period_14_28['D2'].dropna(),
    period_14_28['D3'].dropna(),
    "14-28 days"
)

check_variance_equality(
    period_28_42['D1'].dropna(),
    period_28_42['D2'].dropna(),
    period_28_42['D3'].dropna(),
    "28-42 days"
)

def estimate_params(data):
    mean, std = norm.fit(data)
    shape, loc, scale = lognorm.fit(data, floc=0)
    return {'normal': (mean, std), 'lognormal': (shape, loc, scale)}

def plot_with_distributions(data, group, period):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Observed Data')
    params = estimate_params(data)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    mean, std = params['normal']
    plt.plot(x, norm.pdf(x, mean, std), 'r-', label=f'Normal: μ={mean:.2f}, σ={std:.2f}')

    shape, loc, scale = params['lognormal']
    plt.plot(x, lognorm.pdf(x, shape, loc, scale), 'g--', label=f'Lognormal: Shape={shape:.2f}, Scale={scale:.2f}')

    plt.title(f"Distribution Fit for Group {group} ({period})")
    plt.xlabel("Observed Values")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

for period, data in zip(["14-28 days", "28-42 days"], [period_14_28, period_28_42]):
    for group in ['D1', 'D2', 'D3']:
        group_data = data[group].dropna()
        plot_with_distributions(group_data, group, period)

def bootstrap(data, n_bootstrap=1000):
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    medians = [np.median(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    return {'means': means, 'medians': medians}
bootstrap_results = {}

for period, data in zip(["14-28 days", "28-42 days"], [period_14_28, period_28_42]):
    bootstrap_results[period] = {}
    for group in ['D1', 'D2', 'D3']:
        group_data = data[group].dropna()
        bootstrap_results[period][group] = bootstrap(group_data)


print(bootstrap_results["14-28 days"]["D1"]["means"][:5])

def confidence_intervals(bootstrap_data, confidence_level=95):
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    mean_ci = np.percentile(bootstrap_data['means'], [lower_percentile, upper_percentile])
    median_ci = np.percentile(bootstrap_data['medians'], [lower_percentile, upper_percentile])
    return {'mean_ci': mean_ci, 'median_ci': median_ci}
ci_results = []

for period, groups in bootstrap_results.items():
    for group, stats in groups.items():
        ci = confidence_intervals(stats)
        ci_results.append({
            'Period': period,
            'Group': group,
            'Mean Lower CI': ci['mean_ci'][0],
            'Mean Upper CI': ci['mean_ci'][1],
            'Median Lower CI': ci['median_ci'][0],
            'Median Upper CI': ci['median_ci'][1]
        })


ci_results_df = pd.DataFrame(ci_results)
print(ci_results_df)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

def prepare_long_format(data, period):
    long_data = data.melt(var_name='Group', value_name='Weight', ignore_index=False).dropna()
    long_data['Period'] = period
    return long_data

long_data_14_28 = prepare_long_format(period_14_28, "14-28 days")
long_data_28_42 = prepare_long_format(period_28_42, "28-42 days")


def tukey_hsd_test(data, period):
    print(f"\nTukey's HSD Test for {period}:")
    tukey_result = pairwise_tukeyhsd(endog=data['Weight'], groups=data['Group'], alpha=0.05)
    print(tukey_result)
    return tukey_result

tukey_hsd_test(long_data_14_28, "14-28 days")
tukey_hsd_test(long_data_28_42, "28-42 days")

from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm

def bonferroni_test(data, period):
    print(f"\nBonferroni Test for {period}:")
    mc = MultiComparison(data['Weight'], data['Group'])
    result = mc.allpairtest(sm.stats.ttest_ind, method='b', alpha=0.05)
    print(result[0])
    return result

def scheffe_test(data, period):
    print(f"\nScheffé Test for {period}:")
    mc = MultiComparison(data['Weight'], data['Group'])
    result = mc.allpairtest(lambda x, y: sm.stats.ttest_ind(x, y, usevar='pooled'), method="b")
    print(result[0])
    return result

long_data_14_28 = prepare_long_format(period_14_28, "14-28 days")
long_data_28_42 = prepare_long_format(period_28_42, "28-42 days")

bonferroni_test(long_data_14_28, "14-28 days")
bonferroni_test(long_data_28_42, "28-42 days")

scheffe_test(long_data_14_28, "14-28 days")
scheffe_test(long_data_28_42, "28-42 days")

def calculate_confidence_intervals(data, confidence_level=95):
    means = []
    for _ in range(1000):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - confidence_level) / 2)
    upper = np.percentile(means, 100 - (100 - confidence_level) / 2)
    return lower, upper

ci_results = []

for diet in ['D1', 'D2', 'D3']:
    for period in ['14-28 days', '28-42 days']:
        subset = df_long[(df_long['Diet'] == diet) & (df_long['Period'] == period)]['Weight']
        lower, upper = calculate_confidence_intervals(subset)
        ci_results.append({
            'Diet': diet,
            'Period': period,
            'Mean': np.mean(subset),
            'Lower CI': lower,
            'Upper CI': upper
        })

ci_df = pd.DataFrame(ci_results)
print("Confidence Intervals for Diet and Period combinations:")
print(ci_df)

def mean(data):
    return sum(data) / len(data)

def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:  # if even
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:  # if odd
        return sorted_data[mid]

def mode(data):
    frequency = {}
    for item in data:
        frequency[item] = frequency.get(item, 0) + 1
    return max(frequency, key=frequency.get)


print('Mean:', mean(data), 'Median:', median(data), 'Mode:', mode(data))

import pandas as pd
from statistics import mean, median

mean_manual = mean(data)
median_manual = median(data)
mode_manual = mode(data)

mean_np = np.mean(data)
median_np = np.median(data)
def func_mode(data):
    series = pd.Series(data)
    frequency = series.value_counts()
    return frequency.idxmax()
mode_pd = func_mode(data)

results = pd.DataFrame({
    'Statistic': ['Mean', 'Median', 'Mode'],
    'Manual': [mean_manual, median_manual, mode_manual],
    'NumPy/Pandas': [mean_np, median_np, mode_pd]
})
results

def weighted_mean(values, frequencies):
    return sum(x * n for x, n in zip(values, frequencies)) / sum(frequencies)

def weighted_median(values, frequencies):
    cumulative_frequencies = np.cumsum(frequencies)
    total_count = cumulative_frequencies[-1]
    for i, cum_freq in enumerate(cumulative_frequencies):
        if cum_freq >= total_count / 2:
            return values[i]

def weighted_mode(values, frequencies):
    return values[frequencies.index(max(frequencies))]

def weighted_variance(values, frequencies, mean_value):
    mean_squared = sum((x ** 2) * n for x, n in zip(values, frequencies)) / sum(frequencies)
    return mean_squared - mean_value ** 2

def weighted_std(variance_value):
    return variance_value ** 0.5

mean_weighted = weighted_mean(xi, ni)
median_weighted = weighted_median(xi, ni)
mode_weighted = weighted_mode(xi, ni)
variance_weighted = weighted_variance(xi, ni, mean_weighted)
std_weighted = weighted_std(variance_weighted)

print(f"Mean: {mean_weighted}, Median: {median_weighted}, Mode: {mode_weighted}, Variance: {variance_weighted}, Std: {std_weighted}")

def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    margin_of_error = 1.96 * std_err  # For 95% confidence interval (Z-score 1.96)
    return mean - margin_of_error, mean + margin_of_error

alpha_X = confidence_interval(sample_X)
alpha_Y = confidence_interval(sample_Y)

mean_diff = mean_X - mean_Y
std_diff = np.sqrt((std_X**2 / len(sample_X)) + (std_Y**2 / len(sample_Y)))
alpha_diff = (mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff)

results_XY = pd.DataFrame({
    'Statistic': ['Mean X', 'Mean Y', 'Mean Difference', 'Std X', 'Std Y','Std Difference', '95% Confidence Interval X', '95% Confidence Interva Y', '95% Confidence Interva Difference'],
    'Value': [mean_X, mean_Y, mean_diff, std_X, std_Y, std_diff, alpha_X, alpha_Y, alpha_diff]
})
results_XY

def R(df, col1, col2):
  x = df[col1]
  y = df[col2]
  n = len(x)

  x_mean = sum(x) / n
  y_mean = sum(y) / n

  num = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(n)])
  den = np.sqrt(sum([(x[i] - x_mean)**2 for i in range(n)]) * sum([(y[i] - y_mean)**2 for i in range(n)]))
  return num / den

iris_data = pd.read_csv('iris.csv').drop(columns=['variety'])
pairs = [('sepal.length', 'sepal.width'), ('sepal.length', 'petal.length'),
         ('sepal.length', 'petal.width'), ('sepal.width', 'petal.length'),
         ('sepal.width', 'petal.width'), ('petal.length', 'petal.width')]


results_iris = pd.DataFrame([(col1, col2, R(iris_data, col1, col2)) for col1, col2 in pairs], columns=['col1', 'col2', 'R Manual'])
results_iris['R Theory'] = [iris_data.corr().loc[col1, col2] for col1, col2 in zip(results_iris['col1'], results_iris['col2'])]
results_iris

def CI(r, df, confidence_level=95):
    if confidence_level == 95:
        z_critical = 1.96
    elif confidence_level == 99:
        z_critical = 2.57
    else:
        raise ValueError("Only 95% and 99% confidence levels are supported")

    n = len(df)
    Z = 0.5 * np.log((1 + r) / (1 - r))
    sZ = 1 / np.sqrt(n - 3)
    Zinf = Z - z_critical * sZ
    Zsup = Z + z_critical * sZ
    Rinf = (np.exp(2 * Zinf) - 1) / (np.exp(2 * Zinf) + 1)
    Rsup = (np.exp(2 * Zsup) - 1) / (np.exp(2 * Zsup) + 1)
    return Rinf, Rsup

results_iris['Inf CI (95%)'], results_iris['Sup CI (95%)'] = zip(*[CI(r, iris_data['petal.length']) for r in results_iris['R Manual']])
results_iris['Inf CI (99%)'], results_iris['Sup CI (99%)'] = zip(*[CI(r, iris_data['petal.length'], confidence_level=99) for r in results_iris['R Manual']])
results_iris

# prompt: ### Пример кода для удаления элемента из динамического массива в Python:

import time
import sys
import numpy as np
import statistics

# Инициализация динамического массива (пустого списка)
a = []

# Функция для измерения времени выполнения операции
def time_operation(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time) * 10**9  # Время в наносекундах


# Заполнение массива
for i in range(1000):
    a.append(i)


# Измерение времени удаления элемента с конца массива
time_end = time_operation(a.pop)

# Измерение времени удаления элемента из середины массива
time_middle = time_operation(a.pop, 500)

# Вывод результатов
print(f"Время удаления элемента с конца: {time_end} наносекунд")
print(f"Время удаления элемента из середины: {time_middle} наносекунд")

# Анализ памяти (можно использовать sys.getsizeof(a))
# ... (добавить код для анализа памяти)

import scipy.stats as stats

def normal(data, group, period):
    plt.figure(figsize=(6, 4))

    plt.hist(data, bins=10, density=True, alpha=0.6, color='blue', edgecolor='black', label='Discrete')

    mu, std, median = np.mean(data), np.std(data), np.median(data)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)

    plt.plot(x, p, 'r-', linewidth=2, label=f'Normal \nMean: {mu:.2f}, Std: {std:.2f}, Median: {median:.2f}')

    plt.title(f'{group} during {period}')
    plt.xlabel('Weights')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

for group in ['D1', 'D2', 'D3']:
    for period, data in zip(['14-28 days', '28-42 days'], [period_14_28, period_28_42]):
        normal(data[group].dropna(), group, period)

def lognormal(data, group, period):
    plt.figure(figsize=(6, 4))

    plt.hist(data, bins=10, density=True, alpha=0.6, color='blue', edgecolor='black', label='Discrete')

    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    median = np.median(data)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.lognorm.pdf(x, shape, loc, scale)

    plt.plot(x, p, 'r-', linewidth=2, label=f'Lognormal\nShape: {shape:.2f}, Scale: {scale:.2f}, Median: {median:.2f}')

    plt.title(f'{group} during {period}')
    plt.xlabel('Weights')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

for group in ['D1', 'D2', 'D3']:
    for period, data in zip(['14-28 days', '28-42 days'], [period_14_28, period_28_42]):
        lognormal(data[group].dropna(), group, period)

def CI(bootstrap_data):
    lower_percentile = (100 - 95) / 2
    upper_percentile = 100 - lower_percentile
    mean_ci = np.percentile(bootstrap_data['means'], [lower_percentile, upper_percentile])
    median_ci = np.percentile(bootstrap_data['medians'], [lower_percentile, upper_percentile])
    return mean_ci, median_ci

ci = []
for period, group_results in bootstrap_results.items():
    for group, results in group_results.items():
        mean_ci, median_ci = CI(results)
        ci.append({
            'Period': period,
            'Group': group,
            'Mean Lower CI': mean_ci[0],
            'Mean Upper CI': mean_ci[1],
            'Median Lower CI': median_ci[0],
            'Median Upper CI': median_ci[1]
        })


results_ci = pd.DataFrame(ci)
results_ci
