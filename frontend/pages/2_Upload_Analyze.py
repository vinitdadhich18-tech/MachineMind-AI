"""
2_Upload_Analyze.py - Operational Analysis Workstation: Snapshot Upload, Waveform Preview & Sequence Replay.
"""

import streamlit as st
from frontend.services.api_client import run_inference, list_machines, get_health, ApiError
from frontend.utils.csv_validation import validate_snapshot_csv
from frontend.components.theme import inject_theme
from frontend.components.status import (
    render_global_header,
    render_sidebar_shell,
    render_status_badge,
    render_footer
)
from frontend.components.channel_cards import render_channel_cards
from frontend.components.charts import render_anomaly_score_chart, render_waveform_preview
from frontend.utils.formatting import format_timestamp

st.set_page_config(page_title="MachineMind AI — Analyze", page_icon="📈", layout="wide")


def main():
    inject_theme()

    try:
        health = get_health()
        render_sidebar_shell(health)
        machines = list_machines()
    except ApiError as err:
        st.error(f"Backend Connection Refused: {err.message}")
        st.stop()

    if not machines:
        render_global_header("NO MACHINERY REGISTERED")
        st.warning("No machines found. Please register a machine on the MACHINES page first.")
        render_footer()
        return

    machine_ids = [m["machine_id"] for m in machines]
    selected_id = st.session_state.get("selected_machine_id")
    if selected_id not in machine_ids:
        selected_id = machine_ids[0]
        st.session_state["selected_machine_id"] = selected_id

    # Sidebar Selection
    selected_id = st.sidebar.selectbox(
        "MONITORED EQUIPMENT",
        options=machine_ids,
        index=machine_ids.index(selected_id),
        key="analyze_machine_select"
    )
    st.session_state["selected_machine_id"] = selected_id

    active_m = next((m for m in machines if m["machine_id"] == selected_id), machines[0])
    render_global_header(active_machine_id=active_m.get("name", selected_id))

    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>OPERATIONAL ANALYSIS WORKSTATION</div>", unsafe_allow_html=True)

    mode = st.radio("ANALYSIS MODE", ["SINGLE SNAPSHOT", "SEQUENCE REPLAY"], horizontal=True)

    if mode == "SINGLE SNAPSHOT":
        col_up, col_ts = st.columns([3, 1])

        with col_up:
            uploaded_file = st.file_uploader(
                "Upload 1-Second Vibration Snapshot File (.csv, .txt, .tsv)",
                type=["csv", "txt", "tsv"],
                accept_multiple_files=False,
                help="Raw 4-channel 20,480 samples x 4 columns vibration array"
            )

        with col_ts:
            snapshot_time = st.text_input(
                "Snapshot Timestamp (Optional)",
                placeholder="2004-02-12T10:32:39Z"
            )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            # Client validation
            is_valid, err_msg, summary_details, preview_df = validate_snapshot_csv(file_bytes, filename)

            if not is_valid:
                st.error(f"Validation Error: {err_msg}")
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 4px; padding: 8px 12px; font-size: 0.78rem; color: #10B981; margin-bottom: 12px;">
                        <span class="status-dot-normal"></span> <strong>20,480 samples × 4 channels | VALID SNAPSHOT MATRIX</strong> ({filename})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col_run, _ = st.columns([1, 4])
                with col_run:
                    run_btn = st.button("RUN ANOMALY INFERENCE", type="primary")

                if run_btn:
                    with st.spinner("Processing feature extraction & Isolation Forest scoring..."):
                        try:
                            result = run_inference(
                                machine_id=selected_id,
                                file_bytes=file_bytes,
                                filename=filename,
                                snapshot_time=snapshot_time if snapshot_time.strip() else None
                            )
                            st.session_state["last_inference_result"] = result
                            st.session_state["last_inference_machine"] = selected_id
                        except ApiError as err:
                            st.error(f"Inference Error ({err.code}): {err.message}")

                with st.expander("Raw Signal Waveform & Statistical Bounds", expanded=False):
                    render_waveform_preview(preview_df)
                    if summary_details.get("channel_stats"):
                        st.json(summary_details["channel_stats"])

        # Display Result Console if available
        last_result = st.session_state.get("last_inference_result")
        last_machine = st.session_state.get("last_inference_machine")

        if last_result and last_machine == selected_id:
            st.markdown("---")
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.04em; margin-bottom: 8px;'>INFERENCE ANALYSIS RESULT CONSOLE</div>", unsafe_allow_html=True)

            overall = last_result.get("overall", {})
            overall_state = overall.get("state", "normal")
            label_text = overall.get("label", "")
            pers_conf = overall.get("persistence_confirmed", False)
            channels = last_result.get("channels", [])
            req_k = last_result.get("persistence", {}).get("required_consecutive_snapshots", 3)

            st_dot = "status-dot-normal" if overall_state == "normal" else ("status-dot-watch" if overall_state == "watch" else "status-dot-anomaly")
            st_pill = "pill-normal" if overall_state == "normal" else ("pill-watch" if overall_state == "watch" else "pill-anomaly")

            # Analysis Result Block
            st.markdown(
                f"""
                <div class="ctrl-panel">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 1.1rem; font-weight: 800; color: #F1F5F9;"><span class="{st_dot}"></span>{label_text}</span>
                        <span class="{st_pill}">{overall_state.upper()}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 10px;">
                        Persistence State: <strong style="color: {'#EF4444' if pers_conf else '#F59E0B'};">{'CONFIRMED (3/3 Consecutive)' if pers_conf else 'Awaiting consecutive snapshots'}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Horizontal Condition Strip
            render_channel_cards(channels, required_consecutive=req_k)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px;'>ANOMALY SCORE vs P99 THRESHOLD COMPARISON</div>", unsafe_allow_html=True)
            render_anomaly_score_chart(channels)

    else:
        # Sequence Replay Mode
        st.markdown("<div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 10px;'>Upload chronological NASA IMS sequence files to demonstrate the 3-snapshot persistence rule.</div>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload Sequence Files",
            accept_multiple_files=True
        )

        if uploaded_files:
            sorted_files = sorted(uploaded_files, key=lambda f: f.name)
            st.info(f"Loaded {len(sorted_files)} sequence files.")

            if st.button("EXECUTE SEQUENCE REPLAY", type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                replay_summary = []

                for idx, f_obj in enumerate(sorted_files, start=1):
                    status_text.text(f"Processing snapshot {idx}/{len(sorted_files)} ({f_obj.name})...")
                    f_bytes = f_obj.getvalue()

                    try:
                        res = run_inference(machine_id=selected_id, file_bytes=f_bytes, filename=f_obj.name)
                        ov = res.get("overall", {})
                        alert_data = res.get("alert", {})
                        max_count = max(ch.get("consecutive_flagged_count", 0) for ch in res.get("channels", []))

                        replay_summary.append({
                            "Seq": idx,
                            "Filename": f_obj.name,
                            "State": ov.get("state"),
                            "Label": ov.get("label"),
                            "Max Count": f"{max_count}/3",
                            "Confirmed": "YES" if ov.get("persistence_confirmed") else "No",
                            "Alert": "Created" if alert_data.get("created") else "None"
                        })
                    except ApiError as err:
                        replay_summary.append({"Seq": idx, "Filename": f_obj.name, "State": "error", "Label": err.message})

                    progress_bar.progress(idx / len(sorted_files))

                status_text.text("Sequence Replay Completed.")
                st.dataframe(replay_summary, use_container_width=True, hide_index=True)

    render_footer()


if __name__ == "__main__":
    main()
