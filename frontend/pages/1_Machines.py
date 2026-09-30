"""
1_Machines.py - Equipment Console & Machinery State Management Page.
"""

import streamlit as st
from frontend.services.api_client import list_machines, create_machine, get_machine, get_predictions, get_health, ApiError
from frontend.components.theme import inject_theme
from frontend.components.status import (
    render_global_header,
    render_sidebar_shell,
    render_status_badge,
    render_footer
)
from frontend.components.channel_cards import render_channel_cards
from frontend.components.charts import render_history_chart
from frontend.utils.formatting import format_timestamp

st.set_page_config(page_title="MachineMind AI — Machines", page_icon="⚙️", layout="wide")


def main():
    inject_theme()

    try:
        health = get_health()
        render_sidebar_shell(health)
        machines = list_machines()
    except ApiError as err:
        st.error(f"Failed to connect to backend: {err.message}")
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

    # Sidebar selection
    selected_id = st.sidebar.selectbox(
        "MONITORED EQUIPMENT",
        options=machine_ids,
        index=machine_ids.index(selected_id),
        key="machines_page_select"
    )
    st.session_state["selected_machine_id"] = selected_id

    active_m = next((m for m in machines if m["machine_id"] == selected_id), machines[0])
    render_global_header(active_machine_id=active_m.get("name", selected_id))

    col_list, col_detail = st.columns([1, 2])

    # LEFT: EQUIPMENT LIST CONSOLE (1/3 width)
    with col_list:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>EQUIPMENT CONSOLE LIST</div>", unsafe_allow_html=True)
        for m in machines:
            m_id = m["machine_id"]
            st_code = m.get("status", "no_data")
            is_active = (m_id == selected_id)
            bg_color = "#161D2B" if is_active else "#101520"
            border_color = "#00E5FF" if is_active else "#1C2436"
            dot_class = "status-dot-normal" if st_code == "normal" else ("status-dot-watch" if st_code == "watch" else ("status-dot-anomaly" if st_code == "anomaly_detected" else "status-dot-gray"))
            pill_class = "pill-normal" if st_code == "normal" else ("pill-watch" if st_code == "watch" else ("pill-anomaly" if st_code == "anomaly_detected" else "pill-nodata"))
            st_short = "NORMAL" if st_code == "normal" else ("WATCH" if st_code == "watch" else ("ANOMALY" if st_code == "anomaly_detected" else "NO DATA"))

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; padding: 10px 12px; margin-bottom: 6px; cursor: pointer;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.88rem; font-weight: 700; color: #F1F5F9;"><span class="{dot_class}"></span>{m['name']}</span>
                        <span class="{pill_class}">{st_short}</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 4px;">
                        ID: <code style="color: #94A3B8;">{m_id}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # RIGHT: SELECTED MACHINE DETAIL VIEW (2/3 width)
    with col_detail:
        try:
            m_detail = get_machine(selected_id)
        except ApiError as err:
            st.error(err.message)
            st.stop()

        m_name = m_detail.get("name", selected_id)
        m_status = m_detail.get("status", "no_data")
        open_alerts = m_detail.get("open_alerts_count", 0)
        latest_pred = m_detail.get("latest_prediction") or {}
        overall = latest_pred.get("overall", {})
        channels_data = latest_pred.get("channels", [])

        # Fetch predictions as fallback for channels_data if latest_pred.channels is empty
        try:
            pred_data = get_predictions(selected_id, limit=50)
            predictions = pred_data.get("predictions", [])
            if not channels_data and predictions:
                channels_data = predictions[0].get("channels", [])
        except ApiError:
            predictions = []

        st_dot = "status-dot-normal" if m_status == "normal" else ("status-dot-watch" if m_status == "watch" else ("status-dot-anomaly" if m_status == "anomaly_detected" else "status-dot-gray"))
        st_pill = "pill-normal" if m_status == "normal" else ("pill-watch" if m_status == "watch" else ("pill-anomaly" if m_status == "anomaly_detected" else "pill-nodata"))
        st_label = "NORMAL" if m_status == "normal" else ("WATCH" if m_status == "watch" else ("ANOMALY CONFIRMED" if m_status == "anomaly_detected" else "NO DATA"))

        st.markdown(
            f"""
            <div class="ctrl-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-size: 1.2rem; font-weight: 800; color: #F1F5F9;">{m_name}</span>
                        <span style="font-size: 0.78rem; color: #64748B; margin-left: 8px;">ID: {selected_id}</span>
                    </div>
                    <span class="{st_pill}"><span class="{st_dot}"></span>{st_label}</span>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8;">{m_detail.get('description') or 'No equipment description provided.'}</div>
                <div style="display: flex; gap: 24px; margin-top: 10px; padding-top: 8px; border-top: 1px solid #1C2436; font-size: 0.78rem;">
                    <span>LAST ANALYSIS RUN: <strong style="color: #F1F5F9;">{format_timestamp(m_detail.get('updated_at'))}</strong></span>
                    <span>DATASET SNAPSHOT TIME: <strong style="color: #F1F5F9;">{format_timestamp(latest_pred.get('timestamp'))}</strong></span>
                    <span>OPEN ALERTS: <strong style="color: {'#EF4444' if open_alerts > 0 else '#10B981'};">{open_alerts}</strong></span>
                    <span>CONDITION: <strong style="color: #00E5FF;">{overall.get('label', 'N/A')}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Large Historical Trend Chart
        st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px;'>HISTORICAL ANOMALY SCORE TELEMETRY</div>", unsafe_allow_html=True)
        try:
            pred_data = get_predictions(selected_id, limit=50)
            predictions = pred_data.get("predictions", [])
            if predictions:
                render_history_chart(list(reversed(predictions)))
            else:
                st.info("No prediction telemetry history available for this machine.")
        except ApiError:
            pass

        # Channel Status Strip
        if channels_data:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #F1F5F9; margin: 12px 0 6px 0;'>CHANNEL CONDITION STRIP</div>", unsafe_allow_html=True)
            render_channel_cards(channels_data)

        # Secondary Action: Register Machine Expander
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("+ Register New Equipment Unit", expanded=False):
            with st.form("create_machine_form"):
                new_id = st.text_input("Machine ID (e.g. bearing_set2_rig1)")
                new_name = st.text_input("Machine Name (e.g. NASA Test Rig A)")
                new_desc = st.text_area("Description (optional)")
                submitted = st.form_submit_button("Register Equipment", type="primary")

                if submitted:
                    if not new_id or not new_name:
                        st.error("Please provide both Machine ID and Machine Name.")
                    else:
                        try:
                            created = create_machine(new_id, new_name, new_desc)
                            st.success(f"Registered machine '{created['machine_id']}'!")
                            st.session_state["selected_machine_id"] = created["machine_id"]
                            st.rerun()
                        except ApiError as err:
                            st.error(err.message)

    render_footer()


if __name__ == "__main__":
    main()
