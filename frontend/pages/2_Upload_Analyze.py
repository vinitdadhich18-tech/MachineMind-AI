"""
2_Upload_Analyze.py - Core Feature: CSV Snapshot Upload & Anomaly Analysis.
"""

import streamlit as st
from frontend.services.api_client import run_inference, list_machines, ApiError
from frontend.utils.csv_validation import validate_snapshot_csv
from frontend.components.status import render_status_badge, render_footer, DISCLAIMER_TEXT
from frontend.components.channel_cards import render_channel_cards

st.set_page_config(page_title="MachineMind AI — Upload & Analyze", page_icon="📈", layout="wide")


def main():
    st.title("📈 Upload & Analyze Vibration Snapshot")
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

    # Sidebar selection
    st.sidebar.markdown("### 🏭 Selected Machine")
    selected_id = st.sidebar.selectbox(
        "Machine ID",
        options=machine_ids,
        index=machine_ids.index(selected_id),
        key="upload_machine_select"
    )
    st.session_state["selected_machine_id"] = selected_id

    st.subheader(f"Analyzing Machine: `{selected_id}`")

    col_upload, col_opts = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload 1-Second Vibration Snapshot (.csv, .txt, .tsv)",
            type=["csv", "txt", "tsv"],
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

            with st.expander("Preview snapshot data head & statistics", expanded=False):
                st.dataframe(preview_df.head(10), use_container_width=True)
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

    # Render results from session state if available
    last_result = st.session_state.get("last_inference_result")
    last_machine = st.session_state.get("last_inference_machine")

    if last_result and last_machine == selected_id:
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        overall = last_result.get("overall", {})
        overall_state = overall.get("state", "normal")
        overall_label = overall.get("label", "")

        render_status_badge(overall_state)

        # Alert banner if alert created
        alert_info = last_result.get("alert", {})
        if alert_info.get("created"):
            st.error(f"⚠️ **Alert Created!** Severity: `{alert_info.get('severity')}` (Alert ID: `{alert_info.get('alert_id')}`)")

        st.markdown("#### 4-Channel Anomaly Scores & Threshold Exceedance")
        channels = last_result.get("channels", [])
        req_k = last_result.get("persistence", {}).get("required_consecutive_snapshots", 3)
        render_channel_cards(channels, required_consecutive=req_k)

        with st.expander("Full JSON API Payload Response", expanded=False):
            st.json(last_result)

    render_footer()


if __name__ == "__main__":
    main()
