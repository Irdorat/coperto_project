from pathlib import Path

import pandas as pd


def restore_daily_calendar(
    df: pd.DataFrame,
    closed_dates: set[str] | None = None,
) -> pd.DataFrame:
    # Restore missing calendar dates for each restaurant.

    if closed_dates is None:
        closed_dates = set()

    closed_dates = {
        pd.Timestamp(date)
        for date in closed_dates
    }

    restored_groups = []

    for restaurant_id, group in df.groupby(
        'restaurant_id'
    ):
        group = group.sort_values(
            'date'
        ).copy()

        expected_dates = pd.date_range(
            start=group['date'].min(),
            end=group['date'].max(),
            freq='D',
        )

        calendar = pd.DataFrame({
            'date': expected_dates
        })

        restored = calendar.merge(
            group,
            on='date',
            how='left',
        )

        restored['restaurant_id'] = (
            restored['restaurant_id']
            .fillna(restaurant_id)
            .astype(int)
        )

        missing_days = (
            restored['guests'].isna()
        )

        closed_days = (
            missing_days
            & restored['date'].isin(closed_dates)
        )

        export_failures = (
            missing_days
            & ~closed_days
        )

        restored['data_status'] = 'observed'

        restored.loc[
            closed_days,
            'data_status',
        ] = 'closed'

        restored.loc[
            export_failures,
            'data_status',
        ] = 'missing_export'

        # A confirmed closed day has zero guests
        # and zero revenue.

        restored.loc[
            closed_days,
            'guests',
        ] = 0

        if 'revenue' in restored.columns:
            restored.loc[
                closed_days,
                'revenue',
            ] = 0.0

        if 'missing_sales_count' in restored.columns:
            restored.loc[
                closed_days,
                'missing_sales_count',
            ] = 0

        if 'missing_ratio' in restored.columns:
            restored.loc[
                closed_days,
                'missing_ratio',
            ] = 0.0

        restored_groups.append(restored)

    if not restored_groups:
        raise ValueError(
            'Cannot restore calendar for empty data'
        )

    result = pd.concat(
        restored_groups,
        ignore_index=True,
    )

    result = result.sort_values(
        ['restaurant_id', 'date']
    ).reset_index(drop=True)

    return result


def load_daily_data(
    path: str | Path,
    closed_dates: set[str] | None = None,
) -> pd.DataFrame:
    # Load and validate daily restaurant data.

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f'Data file not found: {path}'
        )

    df = pd.read_csv(
        path,
        parse_dates=['date']
    )

    required_columns = {
        'date',
        'guests',
        'restaurant_id',
        'revenue',
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f'Missing required columns: '
            f'{sorted(missing_columns)}'
        )

    if df['date'].isna().any():
        raise ValueError(
            'Column date contains missing values'
        )

    if df['guests'].isna().any():
        raise ValueError(
            'Column guests contains missing values'
        )

    if df['restaurant_id'].isna().any():
        raise ValueError(
            'Column restaurant_id contains missing values'
        )

    if df.duplicated(
        subset=['restaurant_id', 'date']
    ).any():
        raise ValueError(
            'Duplicate restaurant/date pairs found'
        )

    df = df.sort_values(
        ['restaurant_id', 'date']
    ).reset_index(drop=True)

    df = restore_daily_calendar(
        df=df,
        closed_dates=closed_dates,
    )

    validate_daily_frequency(df)

    return df


def validate_daily_frequency(
    df: pd.DataFrame,
) -> None:
    # Check that each restaurant has a continuous calendar.

    for restaurant_id, group in df.groupby(
        'restaurant_id'
    ):
        expected_dates = pd.date_range(
            start=group['date'].min(),
            end=group['date'].max(),
            freq='D',
        )

        missing_dates = expected_dates.difference(
            group['date']
        )

        if not missing_dates.empty:
            raise ValueError(
                f'Restaurant {restaurant_id} has '
                f'{len(missing_dates)} missing dates. '
                f'First missing date: '
                f'{missing_dates[0].date()}'
            )