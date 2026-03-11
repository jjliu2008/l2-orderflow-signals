from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd


EVENT_MODE_ALL = "all"
EVENT_MODE_EVENT = "event"
EVENT_MODE_NON_EVENT = "non-event"

DIRECTION_MODE_ALL = "all"
DIRECTION_MODE_LONG = "long"
DIRECTION_MODE_SHORT = "short"


@dataclass
class DashboardFilters:
    strategy_name: str | None = None
    variant_names: list[str] | None = None
    date_start: date | None = None
    date_end: date | None = None
    sessions: list[str] | None = None
    event_mode: str = EVENT_MODE_ALL
    direction_mode: str = DIRECTION_MODE_ALL
    months: list[str] | None = None
    weekdays: list[str] | None = None
    cost_override_enabled: bool = False
    cost_override_ticks: float = 0.36
    contract: str = "ES"
    oos_only: bool = False
    exclude_power_hour: bool = False

    def to_cache_key(self) -> tuple[Any, ...]:
        variants = tuple(sorted(self.variant_names or []))
        sessions = tuple(sorted(self.sessions or []))
        months = tuple(sorted(self.months or []))
        weekdays = tuple(sorted(self.weekdays or []))
        return (
            self.strategy_name,
            variants,
            self.date_start.isoformat() if self.date_start else None,
            self.date_end.isoformat() if self.date_end else None,
            sessions,
            self.event_mode,
            self.direction_mode,
            months,
            weekdays,
            bool(self.cost_override_enabled),
            float(self.cost_override_ticks),
            self.contract,
            bool(self.oos_only),
            bool(self.exclude_power_hour),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_filters(filters: DashboardFilters, available: dict[str, list[Any]]) -> DashboardFilters:
    out = DashboardFilters(**filters.as_dict())

    strategies = set(available.get("strategy_name", []))
    variants = set(available.get("variant_name", []))
    sessions = set(available.get("session", []))
    months = set(available.get("month", []))
    weekdays = set(available.get("weekday", []))

    if out.strategy_name not in strategies:
        out.strategy_name = next(iter(strategies), None)

    out.variant_names = [v for v in (out.variant_names or []) if v in variants]
    if not out.variant_names and variants:
        out.variant_names = sorted(list(variants))

    out.sessions = [s for s in (out.sessions or []) if s in sessions]
    if not out.sessions and sessions:
        out.sessions = sorted(list(sessions))

    out.months = [m for m in (out.months or []) if m in months]
    if not out.months and months:
        out.months = sorted(list(months))

    out.weekdays = [w for w in (out.weekdays or []) if w in weekdays]
    if not out.weekdays and weekdays:
        out.weekdays = sorted(list(weekdays))

    if out.event_mode not in {EVENT_MODE_ALL, EVENT_MODE_EVENT, EVENT_MODE_NON_EVENT}:
        out.event_mode = EVENT_MODE_ALL
    if out.direction_mode not in {DIRECTION_MODE_ALL, DIRECTION_MODE_LONG, DIRECTION_MODE_SHORT}:
        out.direction_mode = DIRECTION_MODE_ALL
    if out.contract not in {"ES", "MES"}:
        out.contract = "ES"

    if out.cost_override_ticks < 0:
        out.cost_override_ticks = 0.0

    return out


def apply_trade_filters(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    mask = pd.Series(True, index=out.index)

    if filters.strategy_name:
        mask &= out["strategy_name"] == filters.strategy_name
    if filters.variant_names:
        mask &= out["variant_name"].isin(filters.variant_names)

    if filters.date_start is not None:
        mask &= pd.to_datetime(out["date"]).dt.date >= filters.date_start
    if filters.date_end is not None:
        mask &= pd.to_datetime(out["date"]).dt.date <= filters.date_end

    if filters.sessions:
        mask &= out["session"].isin(filters.sessions)
    if filters.exclude_power_hour:
        mask &= out["session"] != "power_hour"

    if filters.event_mode == EVENT_MODE_EVENT:
        mask &= out["is_event_day"] == True  # noqa: E712
    elif filters.event_mode == EVENT_MODE_NON_EVENT:
        mask &= out["is_event_day"] == False  # noqa: E712

    if filters.direction_mode == DIRECTION_MODE_LONG:
        mask &= out["direction"].str.lower() == "long"
    elif filters.direction_mode == DIRECTION_MODE_SHORT:
        mask &= out["direction"].str.lower() == "short"

    if filters.months:
        mask &= out["month"].isin(filters.months)
    if filters.weekdays:
        mask &= out["weekday"].isin(filters.weekdays)

    if filters.oos_only and "oos_flag" in out.columns:
        mask &= out["oos_flag"] == True  # noqa: E712

    return out.loc[mask].copy()


def apply_feature_filters(features_df: pd.DataFrame, trade_ids: pd.Series | list[str]) -> pd.DataFrame:
    if features_df.empty:
        return features_df
    trade_ids = pd.Series(trade_ids).dropna().unique().tolist()
    if not trade_ids:
        return features_df.iloc[0:0].copy()
    return features_df[features_df["trade_id"].isin(trade_ids)].copy()


def build_filters_from_sidebar(
    st,
    available: dict[str, list[Any]],
    key_prefix: str = "global_filter",
) -> DashboardFilters:
    strategies = available.get("strategy_name", [])
    variants = available.get("variant_name", [])
    sessions = available.get("session", [])
    months = available.get("month", [])
    weekdays = available.get("weekday", [])

    st.sidebar.subheader("Global Filters")

    strategy = st.sidebar.selectbox(
        "Strategy",
        options=strategies or [None],
        index=0,
        key=f"{key_prefix}_strategy",
    )
    selected_variants = st.sidebar.multiselect(
        "Variant",
        options=variants,
        default=variants,
        key=f"{key_prefix}_variants",
    )

    date_start = available.get("date_min", [None])[0]
    date_end = available.get("date_max", [None])[0]
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(date_start, date_end) if date_start and date_end else (),
        key=f"{key_prefix}_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
    else:
        d0, d1 = date_start, date_end

    selected_sessions = st.sidebar.multiselect(
        "Session",
        options=sessions,
        default=sessions,
        key=f"{key_prefix}_sessions",
    )

    event_mode = st.sidebar.selectbox(
        "Event Day",
        options=[EVENT_MODE_ALL, EVENT_MODE_EVENT, EVENT_MODE_NON_EVENT],
        index=0,
        key=f"{key_prefix}_event_mode",
    )
    direction_mode = st.sidebar.selectbox(
        "Direction",
        options=[DIRECTION_MODE_ALL, DIRECTION_MODE_LONG, DIRECTION_MODE_SHORT],
        index=0,
        key=f"{key_prefix}_direction_mode",
    )

    selected_months = st.sidebar.multiselect(
        "Month",
        options=months,
        default=months,
        key=f"{key_prefix}_months",
    )
    selected_weekdays = st.sidebar.multiselect(
        "Weekday",
        options=weekdays,
        default=weekdays,
        key=f"{key_prefix}_weekdays",
    )

    cost_override_enabled = st.sidebar.checkbox(
        "Cost Override",
        value=False,
        key=f"{key_prefix}_cost_enabled",
    )
    cost_override_ticks = st.sidebar.number_input(
        "Cost (ticks)",
        value=0.36,
        min_value=0.0,
        step=0.01,
        key=f"{key_prefix}_cost_ticks",
        disabled=not cost_override_enabled,
    )
    contract = st.sidebar.selectbox(
        "Contract",
        options=["ES", "MES"],
        index=0,
        key=f"{key_prefix}_contract",
    )

    oos_only = st.sidebar.checkbox(
        "OOS Only",
        value=False,
        key=f"{key_prefix}_oos_only",
    )
    exclude_power_hour = st.sidebar.checkbox(
        "Exclude Power Hour",
        value=False,
        key=f"{key_prefix}_exclude_power_hour",
    )

    filters = DashboardFilters(
        strategy_name=strategy,
        variant_names=selected_variants,
        date_start=d0,
        date_end=d1,
        sessions=selected_sessions,
        event_mode=event_mode,
        direction_mode=direction_mode,
        months=selected_months,
        weekdays=selected_weekdays,
        cost_override_enabled=cost_override_enabled,
        cost_override_ticks=float(cost_override_ticks),
        contract=contract,
        oos_only=oos_only,
        exclude_power_hour=exclude_power_hour,
    )
    return sanitize_filters(filters, available)
