"""
3_History.py - Prediction History & Score Trends Console.
"""

import streamlit as st
from frontend.services.api_client import list_machines, get_predictions, get_health, ApiError
from frontend.components.theme import inject_theme
from frontend.components.status import (
    render_global_header,
    render_sidebar_shell,
    render_status_badge,
    render_footer
)
from frontend.components.charts import render_history_chart
from frontend.utils.formatting import format_timestamp

st.set_page_config(page_title="MachineMind AI — History", page_icon="📜", layout="wide")


def main():
    inject_theme()

    try:
        health = get_health()
        render_sidebar_shell(health)
        machines = list_machines()
    except ApiError as err:
        st.error(f"Backend Connection Error: {err.message}")
        st.stop()

    if not machines:
        render_global_header("NO MACHINERY REGISTERED")
        st.warning("No machines found.")
        render_footer()
        return

    machine_ids = [m["machine_id"] for m in machines]
    selected_id = st.session_state.get("selected_machine_id")
    if selected_id not in machine_ids:
        selected_id = machine_ids[0]
        st.session_state["selected_machine_id"] = selected_id

    # Sidebar Selection & Depth Slider
    selected_id = st.sidebar.selectbox(
        "MONITORED EQUIPMENT",
        options=machine_ids,
        index=machine_ids.index(selected_id),
        key="history_machine_select"
    )
    st.session_state["selected_machine_id"] = selected_id

    limit = st.sidebar.slider("TELEMETRY DEPTH (SNAPSHOTS)", min_value=10, max_value=200, value=50, step=10)

    try:
        data = get_predictions(selected_id, limit=limit)
    except ApiError as err:
        st.error(f"Failed to load prediction history: {err.message}")
        st.stop()

    predictions = data.get("predictions", [])
    total_count = data.get("total", 0)

    active_m = next((m for m in machines if m["machine_id"] == selected_id), machines[0])
    render_global_header(active_machine_id=active_m.get("name", selected_id))

    st.markdown(
        f"""
        <div style="font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;">
            FULL-WIDTH ANOMALY SCORE TELEMETRY TIMELINE ({len(predictions)} displayed / {total_count} total)
        </div>
        """,
        unsafe_allow_html=True
    )

    if not predictions:
        st.info("No historical telemetry recorded for this machine yet.")
    else:
        # 1. Full-Width Anomaly Score Timeline Chart
        render_history_chart(list(reversed(predictions)))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>TELEMETRY RECORD LOG TABLE</div>", unsafe_allow_html=True)

        table_rows = []
        for pred in predictions:
            row = {
                "Dataset Snapshot Time": format_timestamp(pred.get("timestamp")),
                "System State": pred.get("overall", {}).get("state", "no_data").upper(),
                "Overall Label": pred.get("overall", {}).get("label"),
                "Filename": pred.get("source_filename", "N/A"),
            }
            for ch in pred.get("channels", []):
                ch_idx = ch.get("channel")
                row[f"CH{ch_idx}"] = f"{ch.get('anomaly_score', 0.0):.4f}"
            table_rows.append(row)

        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    render_footer()


if __name__ == "__main__":
    main()
