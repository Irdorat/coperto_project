import pandas as pd
import pytest

from src.features import make_features


def create_test_data() -> pd.DataFrame:
    # Create simple daily data for feature tests.

    dates = pd.date_range(
        start='2024-01-01',
        periods=40,
        freq='D',
    )

    guests = list(range(1, 41))

    return pd.DataFrame({
        'date': dates,
        'restaurant_id': 1,
        'guests': guests,
    })


def test_lag_1() -> None:
    # lag_1 must contain yesterday's target.

    df = create_test_data()

    featured = make_features(df)

    # Day 10 has guests = 10.
    # Its lag_1 must contain guests from day 9.
    assert featured.loc[9, 'guests_lag_1'] == 9


def test_lag_7() -> None:
    #lag_7 must contain target from 7 days ago.

    df = create_test_data()

    featured = make_features(df)

    # Index 10 -> guests = 11.
    # Seven rows earlier -> index 3 -> guests = 4.
    assert featured.loc[10,'guests_lag_7'] == 4


def test_lag_14() -> None:
    # lag_14 must contain target from 14 days ago.

    df = create_test_data()

    featured = make_features(df)

    # Index 20 -> guests = 21.
    # 14 rows earlier -> index 6 -> guests = 7.
    assert featured.loc[20, 'guests_lag_14'] == 7


def test_lag_28() -> None:
    # lag_28 must contain target from 28 days ago.

    df = create_test_data()

    featured = make_features(df)

    # Index 30 -> guests = 31.
    # 28 rows earlier -> index 2 -> guests = 3.
    assert featured.loc[30, 'guests_lag_28'] == 3


def test_rolling_mean_7() -> None:
    # Rolling mean must use only previous observations.

    df = create_test_data()

    featured = make_features(df)

    # At index 7 current guests = 8.
    # Rolling mean must use:
    # 1, 2, 3, 4, 5, 6, 7
    expected = (
        1 + 2 + 3 + 4 + 5 + 6 + 7
    ) / 7

    assert featured.loc[7,'guests_mean_7',] == pytest.approx(expected)


def test_rolling_mean_does_not_use_current_target() -> None:
    # Changing today's target must not change today's rolling feature.

    df = create_test_data()

    featured_before = make_features(df)

    original_mean = featured_before.loc[20,'guests_mean_7']

    # Drastically change the CURRENT target.
    df.loc[20, 'guests'] = 100000

    featured_after = make_features(df)

    changed_mean = featured_after.loc[20, 'guests_mean_7']

    assert original_mean == changed_mean


def test_lag_does_not_use_current_target() -> None:
    # Changing today's target must not change today's lag features.

    df = create_test_data()

    featured_before = make_features(df)

    original_lag = featured_before.loc[20, 'guests_lag_1']

    df.loc[20, 'guests_lag_1'] = 100000

    featured_after = make_features(df)

    changed_lag = featured_after.loc[20, 'guests_lag_1']

    assert original_lag == changed_lag


def test_future_target_does_not_change_past_features() -> None:
    # Changing a future target must not change features of previous dates.

    df = create_test_data()

    featured_before = make_features(df)

    old_lag = featured_before.loc[30, 'guests_lag_7']

    old_mean = featured_before.loc[30, 'guests_mean_7']

    # Change a FUTURE observation.
    df.loc[35, 'guests'] = 100000

    featured_after = make_features(df)

    new_lag = featured_after.loc[30, 'guests_lag_7']

    new_mean = featured_after.loc[30, 'guests_mean_7']

    assert old_lag == new_lag
    assert old_mean == new_mean


def test_first_lag_values_are_missing() -> None:
    # Early rows must be NaN when history is insufficient.

    df = create_test_data()

    featured = make_features(df)

    assert pd.isna(featured.loc[0, 'guests_lag_1'])

    assert pd.isna(featured.loc[0, 'guests_lag_7'])

    assert pd.isna(featured.loc[0, 'guests_lag_28'])


def test_first_28_day_mean_values_are_missing() -> None:
    # 28-day rolling mean requires 28 previous observations.

    df = create_test_data()

    featured = make_features(df)

    for index in range(28):
        assert pd.isna(featured.loc[index,'guests_mean_28'])

    assert not pd.isna(featured.loc[28,'guests_mean_28'])