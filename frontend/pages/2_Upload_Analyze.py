"""
2_Upload_Analyze.py - Core Feature: Snapshot Upload, Waveform Preview & Sequence Replay Analysis.
"""

import streamlit as st
from frontend.services.api_client import run_inference, list_machines, ApiError
from frontend.utils.csv_validation import validate_snapshot_csv
from frontend.components.status import render_status_badge, render_footer, DISCLAIMER_TEXT
from frontend.components.channel_cards import render_channel_cards
from frontend.components.charts import render_anomaly_score_chart, render_waveform_preview

st.set_page_config(page_title="MachineMind AI — Upload & Analyze", page_icon="📈", layout="wide")


def main():
    st.title("📈 Upload & Analyze Vibration Snapshots")
    st.caption(DISCLAIMER_TEXT)
    st.markdown("---")

    # Fetch machines
    try:
        machines = list_machines()
    except ApiError as err:
        st.error(f"⚠️ Backend connection issue: {err.message}")
        st.stop()

    if not machines:
        st.warning("⚠️ No registered machines found. Please create a machine on the Machines page first.")
        st.stop()

    machine_ids = [m["machine_id"] for m in machines]
    selected_id = st.session_state.get("selected_machine_id")

    if selected_id not in machine_ids:
        selected_id = machine_ids[0]
        st.session_state["selected_machine_id"] = selected_id

    # Sidebar selection & mode toggle
    st.sidebar.markdown("### 🏭 Selected Machine")
    selected_id = st.sidebar.selectbox(
        "Machine ID",
        options=machine_ids,
        index=machine_ids.index(selected_id),
        key="upload_machine_select"
    )
    st.session_state["selected_machine_id"] = selected_id

    mode = st.radio("Analysis Mode", ["Single Snapshot Analysis", "Sequence Replay Mode (Multiple Files)"], horizontal=True)

    st.subheader(f"Analyzing Machine: `{selected_id}`")

    if mode == "Single Snapshot Analysis":
        col_upload, col_opts = st.columns([3, 2])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload 1-Second Vibration Snapshot (.csv, .txt, .tsv)",
                type=["csv", "txt", "tsv"],
                accept_multiple_files=False,
                help="Upload raw 4-channel vibration file of exactly 20480 samples x 4 columns."
            )

        with col_opts:
            snapshot_time = st.text_input(
                "Snapshot Timestamp (optional ISO-8601 string)",
                placeholder="e.g. 2004-02-12T10:32:39Z",
                help="Defaults to server UTC time if left blank."
            )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            # Client-side pre-validation
            is_valid, err_msg, summary_details, preview_df = validate_snapshot_csv(file_bytes, filename)

            if not is_valid:
                st.error(f"❌ Client Validation Error: {err_msg}")
                if summary_details:
                    with st.expander("Validation details"):
                        st.json(summary_details)
            else:
                st.success(f"✅ File pre-validated cleanly: **20,480 samples × 4 channels** ({filename})")

                with st.expander("Preview waveform & statistics", expanded=True):
                    st.markdown("##### 4-Channel Raw Vibration Waveform (Downsampled)")
                    render_waveform_preview(preview_df)
                    if summary_details.get("channel_stats"):
                        st.write("**Per-Channel Raw Signal Bounds:**")
                        st.json(summary_details["channel_stats"])

                st.markdown("---")
                run_btn = st.button("🚀 Run Anomaly Inference", type="primary")

                if run_btn:
                    with st.spinner("Processing DC mean-centering, 28 time-domain features, and Isolation Forest scoring..."):
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
                            st.error(f"❌ Inference Error ({err.code}): {err.message}")
                            if err.details:
                                with st.expander("Technical Error Details"):
                                    st.json(err.details)

        # Render single result from session state if available
        last_result = st.session_state.get("last_inference_result")
        last_machine = st.session_state.get("last_inference_machine")

        if last_result and last_machine == selected_id:
            st.markdown("---")
            st.subheader("📊 Analysis Results")

            overall = last_result.get("overall", {})
            overall_state = overall.get("state", "normal")

            render_status_badge(overall_state)

            alert_info = last_result.get("alert", {})
            if alert_info.get("created"):
                st.error(f"⚠️ **Alert Created!** Severity: `{alert_info.get('severity')}` (Alert ID: `{alert_info.get('alert_id')}`)")

            st.markdown("#### 4-Channel Anomaly Scores & Threshold Exceedance")
            channels = last_result.get("channels", [])
            req_k = last_result.get("persistence", {}).get("required_consecutive_snapshots", 3)
            render_channel_cards(channels, required_consecutive=req_k)

            st.markdown("#### Score vs P99 Threshold Comparison")
            render_anomaly_score_chart(channels)

            with st.expander("Full JSON API Payload Response", expanded=False):
                st.json(last_result)

    else:
        # Sequence Replay Mode
        st.markdown("#### Chronological Sequence Replay")
        st.caption("Upload multiple snapshot files. They will be sorted chronologically by filename and sent sequentially to demonstrate the 3-snapshot persistence rule.")

        uploaded_files = st.file_uploader(
            "Upload Chronological Sequence Files",
            accept_multiple_files=True,
            help="Upload multiple chronological NASA IMS files (e.g. 2004.02.12.10.32.39, 2004.02.12.10.42.39, 2004.02.12.10.52.39)"
        )

        if uploaded_files:
            # Sort chronologically by filename
            sorted_files = sorted(uploaded_files, key=lambda f: f.name)
            st.info(f"Loaded {len(sorted_files)} files in chronological order: {', '.join([f.name for f in sorted_files[:5]])}{'...' if len(sorted_files) > 5 else ''}")

            if st.button("▶️ Execute Sequence Replay", type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                replay_summary = []

                for idx, f_obj in enumerate(sorted_files, start=1):
                    status_text.text(f"Processing snapshot {idx} of {len(sorted_files)} ({f_obj.name})...")
                    f_bytes = f_obj.getvalue()

                    try:
                        res = run_inference(
                            machine_id=selected_id,
                            file_bytes=f_bytes,
                            filename=f_obj.name
                        )
                        ov = res.get("overall", {})
                        alert_data = res.get("alert", {})
                        max_count = max(ch.get("consecutive_flagged_count", 0) for ch in res.get("channels", []))

                        replay_summary.append({
                            "Seq #": idx,
                            "Filename": f_obj.name,
                            "Overall State": ov.get("state"),
                            "Label": ov.get("label"),
                            "Snapshot Flagged": "🚩 Yes" if ov.get("snapshot_flagged") else "✅ No",
                            "Max Consecutive Count": f"{max_count} / 3",
                            "Persistence Confirmed": "🚨 YES" if ov.get("persistence_confirmed") else "No",
                            "Alert Triggered": "⚠️ Created" if alert_data.get("created") else "None"
                        })
                    except ApiError as err:
                        replay_summary.append({
                            "Seq #": idx,
                            "Filename": f_obj.name,
                            "Overall State": "error",
                            "Label": f"Error: {err.message}",
                            "Snapshot Flagged": "N/A",
                            "Max Consecutive Count": "N/A",
                            "Persistence Confirmed": "N/A",
                            "Alert Triggered": "N/A"
                        })

                    progress_bar.progress(idx / len(sorted_files))

                status_text.text("✅ Sequence Replay Completed!")
                st.markdown("#### Sequence Replay Results Summary")
                st.dataframe(replay_summary, use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()
