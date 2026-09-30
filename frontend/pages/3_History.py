"""
3_History.py - Prediction History & Time-Series Score Visualization Page.
"""

import streamlit as st
from frontend.services.api_client import list_machines, get_predictions, ApiError
from frontend.components.status import render_status_badge, render_footer, DISCLAIMER_TEXT
from frontend.components.charts import render_history_chart

st.set_page_config(page_title="MachineMind AI — History", page_icon="📜", layout="wide")


def main():
    st.title("📜 Prediction History & Score Trends")
    st.caption(DISCLAIMER_TEXT)
    st.markdown("---")

    try:
        machines = list_machines()
    except ApiError as err:
        st.error(f"⚠️ Backend connection issue: {err.message}")
        st.stop()

    if not machines:
        st.warning("No machines registered yet. Please create a machine first.")
        st.stop()

    machine_ids = [m["machine_id"] for m in machines]
    selected_id = st.session_state.get("selected_machine_id")

    if selected_id not in machine_ids:
        selected_id = machine_ids[0]
        st.session_state["selected_machine_id"] = selected_id

    st.sidebar.markdown("### 🏭 Selected Machine")
    selected_id = st.sidebar.selectbox(
        "Machine ID",
        options=machine_ids,
        index=machine_ids.index(selected_id),
        key="history_machine_select"
    )
    st.session_state["selected_machine_id"] = selected_id

    limit = st.slider("History Depth (Max Snapshots)", min_value=10, max_value=200, value=50, step=10)

    try:
        data = get_predictions(selected_id, limit=limit)
    except ApiError as err:
        st.error(f"Failed to load prediction history: {err.message}")
        st.stop()

    predictions = data.get("predictions", [])
    total_count = data.get("total", 0)

    st.subheader(f"History for `{selected_id}` ({len(predictions)} shown / {total_count} total)")

    if not predictions:
        st.info("No prediction history recorded for this machine yet. Upload a snapshot on **2_Upload_Analyze** to begin.")
    else:
        st.markdown("#### Per-Channel Anomaly Score Trend Over Time")
        # Reverse predictions list for chronological left-to-right plotting
        render_history_chart(list(reversed(predictions)))

        st.markdown("#### Detailed Prediction Records")
        table_rows = []
        for pred in predictions:
            row = {
                "Timestamp": pred.get("timestamp", "")[:19],
                "Overall Label": pred.get("overall", {}).get("label"),
                "State": pred.get("overall", {}).get("state"),
                "Filename": pred.get("source_filename", "N/A"),
                "Model Version": pred.get("model_version", "v1")
            }
            # Add per channel score and flag columns
            for ch in pred.get("channels", []):
                ch_idx = ch.get("channel")
                row[f"Ch{ch_idx} Score"] = f"{ch.get('anomaly_score', 0.0):.4f}"
                row[f"Ch{ch_idx} Flag"] = "🚩" if ch.get("snapshot_flagged") else "✅"

            table_rows.append(row)

        st.dataframe(table_rows, use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()
