"""
1_Machines.py - Machine Selection & Management Page.
"""

import streamlit as st
from frontend.services.api_client import list_machines, create_machine, get_machine, ApiError
from frontend.components.status import render_status_badge, render_footer, DISCLAIMER_TEXT

st.set_page_config(page_title="MachineMind AI — Machines", page_icon="🏭", layout="wide")


def render_sidebar_machine_selector(machines):
    """Renders machine selector in sidebar and persists selection to session_state."""
    st.sidebar.markdown("### 🏭 Selected Machine")
    if not machines:
        st.sidebar.info("No machines registered yet.")
        st.session_state["selected_machine_id"] = None
        return None

    machine_ids = [m["machine_id"] for m in machines]

    current_sel = st.session_state.get("selected_machine_id")
    default_idx = machine_ids.index(current_sel) if current_sel in machine_ids else 0

    selected_id = st.sidebar.selectbox(
        "Select Machine",
        options=machine_ids,
        index=default_idx,
        key="sidebar_machine_select"
    )

    st.session_state["selected_machine_id"] = selected_id
    return selected_id


def main():
    st.title("🏭 Machinery Registry & Selection")
    st.caption(DISCLAIMER_TEXT)
    st.markdown("---")

    # Fetch machines from API
    try:
        machines = list_machines()
    except ApiError as err:
        st.error(f"⚠️ Failed to connect to backend: {err.message}")
        if err.details:
            with st.expander("Technical details"):
                st.json(err.details)
        st.stop()

    selected_id = render_sidebar_machine_selector(machines)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Registered Machinery")
        if not machines:
            st.warning("No machines found. Please register a machine using the form on the right.")
        else:
            table_data = []
            for m in machines:
                table_data.append({
                    "Machine ID": m["machine_id"],
                    "Name": m["name"],
                    "Status": m.get("status", "no_data"),
                    "Created At": m.get("created_at", "")[:19]
                })
            st.dataframe(table_data, use_container_width=True)

            if selected_id:
                st.markdown("---")
                st.subheader(f"Selected Machine: {selected_id}")
                try:
                    m_detail = get_machine(selected_id)
                    st.write(f"**Name:** {m_detail['name']}")
                    st.write(f"**Description:** {m_detail.get('description', 'N/A')}")
                    st.write(f"**Open Alerts:** {m_detail.get('open_alerts_count', 0)}")
                    render_status_badge(m_detail.get("status", "no_data"))

                    latest_pred = m_detail.get("latest_prediction")
                    if latest_pred:
                        st.caption(f"Latest prediction timestamp: {latest_pred.get('timestamp')}")
                        st.write(f"Latest status label: **{latest_pred.get('overall', {}).get('label')}**")
                except ApiError as err:
                    st.error(err.message)

    with col_right:
        st.subheader("➕ Register New Machine")
        with st.form("create_machine_form"):
            new_id = st.text_input("Machine ID (e.g. bearing_set2_rig1)", help="1-64 chars: A-Z, a-z, 0-9, _, -")
            new_name = st.text_input("Machine Name (e.g. NASA Test Rig A)")
            new_desc = st.text_area("Description (optional)", help="Context or location details")
            submitted = st.form_submit_button("Create Machine")

            if submitted:
                if not new_id or not new_name:
                    st.error("Please provide both Machine ID and Machine Name.")
                else:
                    try:
                        created = create_machine(new_id, new_name, new_desc)
                        st.success(f"✅ Machine '{created['machine_id']}' created successfully!")
                        st.session_state["selected_machine_id"] = created["machine_id"]
                        st.rerun()
                    except ApiError as err:
                        st.error(f"❌ {err.message}")
                        if err.details:
                            with st.expander("Details"):
                                st.json(err.details)

    render_footer()


if __name__ == "__main__":
    main()
