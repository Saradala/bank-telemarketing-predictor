import pandas as pd

from src.preprocessing import find_high_correlations


def test_find_high_correlations_ignores_non_numeric_selected_columns():
    """Explicit column selection should ignore non-numeric columns."""
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [2, 4, 6, 8],
            "category": ["a", "b", "c", "d"],
        }
    )

    result = find_high_correlations(
        data,
        columns=["feature_a", "feature_b", "category"],
        threshold=0.80,
    )

    assert len(result) == 1
    assert result.iloc[0]["feature_1"] == "feature_a"
    assert result.iloc[0]["feature_2"] == "feature_b"