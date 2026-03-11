from __future__ import annotations


def render_table(st, df, title: str, use_container_width: bool = True, hide_index: bool = True) -> None:
    st.subheader(title)
    st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)

