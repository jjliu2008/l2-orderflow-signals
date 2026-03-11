from __future__ import annotations

from research_dashboard.services import filters as flt, loader


def get_global_filters(st, key_prefix: str = "global_filter") -> flt.DashboardFilters:
    options = loader.get_filter_options()
    f = flt.build_filters_from_sidebar(st, options, key_prefix=key_prefix)
    st.session_state["global_filters"] = f.as_dict()
    return f
