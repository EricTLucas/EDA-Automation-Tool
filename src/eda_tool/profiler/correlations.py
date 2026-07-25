import math
import numpy as np
import pandas as pd

from .base import ProfilerComponent, SectionResult, register_component

class CorrelationsComponent(ProfilerComponent):

    name = "correlations"

    def compute(self, df, profile):

        all_cols = profile["columns"].data

        corr_table = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

        for col1 in all_cols:
            for col2 in all_cols:
                corr_table.loc[col1, col2] = universal_corr(df[col1], df[col2])

        return SectionResult(
            name = self.name,
            data = corr_table
            )

def pointbiserialr(binary, numeric):
    """
    Pure-Python/SciPy-free implementation of pointbiserialr.
    binary: array-like of 0/1 values
    numeric: array-like of floats
    Returns: (r, p_value)
    """
    binary = np.asarray(binary)
    numeric = np.asarray(numeric)

    if set(np.unique(binary)) - {0, 1}:
        raise ValueError("Binary variable must contain only 0 and 1.")

    # Means of the two groups
    m1 = numeric[binary == 1].mean()
    m0 = numeric[binary == 0].mean()

    # Proportions
    p = (binary == 1).mean()
    q = 1 - p

    # Standard deviation of numeric
    s = numeric.std(ddof=1)

    # Point biserial correlation
    r = (m1 - m0) * math.sqrt(p * q) / s

    # Compute t statistic
    n = len(numeric)
    t = r * math.sqrt((n - 2) / (1 - r**2))

    # Two-sided p-value from t distribution
    # Using survival function approximation
    # (SciPy-free Student-t CDF approximation)
    def student_t_sf(t, df):
        # Abramowitz-Stegun approximation
        x = abs(t)
        a = 1 / (1 + x / math.sqrt(df))
        return a**df

    p_value = 2 * student_t_sf(t, n - 2)

    return r, p_value

def chi2_contingency(table):
    """
    Pure-Python/SciPy-free chi-square contingency test.
    table: 2D array-like (contingency table)
    Returns: (chi2, p_value, dof, expected)
    """
    table = np.asarray(table, dtype=float)
    rows, cols = table.shape

    # Row/column sums
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    total = table.sum()

    # Expected frequencies
    expected = np.outer(row_sums, col_sums) / total

    # Chi-square statistic
    chi2 = ((table - expected)**2 / expected).sum()

    # Degrees of freedom
    dof = (rows - 1) * (cols - 1)

    # p-value using chi-square survival function approximation
    def chi2_sf(x, k):
        # Using incomplete gamma approximation
        return math.exp(-0.5 * x) * sum((0.5 * x)**i / math.factorial(i) for i in range(k))

    p_value = chi2_sf(chi2, dof)

    return chi2, p_value, dof, expected


def cramers_v(x, y):
    """Cramér's V for categorical-categorical."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def correlation_ratio(categories, values):
    """Correlation ratio (eta) for categorical-numeric."""
    cat = pd.Categorical(categories)
    groups = [values[cat == c] for c in cat.categories]
    means = np.array([g.mean() for g in groups])
    sizes = np.array([len(g) for g in groups])
    overall_mean = values.mean()

    numerator = np.sum(sizes * (means - overall_mean)**2)
    denominator = np.sum((values - overall_mean)**2)
    return np.sqrt(numerator / denominator) if denominator != 0 else 0.0


def universal_corr(x, y):
    """Universal correlation function for any dtype combination."""

    # Drop NA
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    x = df["x"]
    y = df["y"]

    # Identify types
    x_num = pd.api.types.is_numeric_dtype(x)
    y_num = pd.api.types.is_numeric_dtype(y)

    # Case 1: numeric × numeric → Pearson
    if x_num and y_num:
        return x.corr(y)

    # Case 2: numeric × categorical
    if x_num and not y_num:
        if y.nunique() == 2:
            encoded = pd.factorize(y)[0]
            corr, _ = pointbiserialr(encoded, x)
            return corr
        else:
            return correlation_ratio(y, x)

    # Case 3: categorical × numeric
    if not x_num and y_num:
        if x.nunique() == 2:
            encoded = pd.factorize(x)[0]
            corr, _ = pointbiserialr(encoded, y)
            return corr
        else:
            return correlation_ratio(x, y)

    # Case 4: categorical × categorical → Cramér's V
    return cramers_v(x, y)



register_component(CorrelationsComponent())