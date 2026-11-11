"""
kanban_module.py
Interactive Sponsor Vetting & Child Welfare Workflow Kanban Board.
Renders a 3-column operational board directly inside the Streamlit dashboard
for tracking individual case progression, aging flags, and sponsor handoff readiness.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

KANBAN_COLUMNS = ["Vetting Queue", "Home Study Pending", "Ready for Discharge"]

SAMPLE_CASES = [
    {
        "case_id": "UC-2025-0104",
        "child_initials": "J.M. (Age 14)",
        "sponsor_name": "Elena Morales",
        "relationship": "Category 1 (Mother)",
        "days_in_care": 12,
        "priority": "High",
        "facility": "San Antonio Reception Shelter",
        "status": "Vetting Queue"
    },
    {
        "case_id": "UC-2025-0089",
        "child_initials": "A.R. (Age 9)",
        "sponsor_name": "Carlos Rivera",
        "relationship": "Category 1 (Father)",
        "days_in_care": 18,
        "priority": "High",
        "facility": "El Paso Transitional Care",
        "status": "Vetting Queue"
    },
    {
        "case_id": "UC-2025-0062",
        "child_initials": "M.S. (Age 16)",
        "sponsor_name": "Rosa Sanchez",
        "relationship": "Category 2 (Aunt)",
        "days_in_care": 34, # Exceeds 30 days!
        "priority": "Urgent",
        "facility": "Phoenix Youth Center",
        "status": "Home Study Pending"
    },
    {
        "case_id": "UC-2025-0077",
        "child_initials": "L.G. (Age 11)",
        "sponsor_name": "Miguel Gomez",
        "relationship": "Category 2 (Adult Brother)",
        "days_in_care": 25,
        "priority": "Standard",
        "facility": "Houston Care Facility",
        "status": "Home Study Pending"
    },
    {
        "case_id": "UC-2025-0041",
        "child_initials": "D.P. (Age 7)",
        "sponsor_name": "Juana Perez",
        "relationship": "Category 1 (Mother)",
        "days_in_care": 22,
        "priority": "High",
        "facility": "San Diego Shelter Hub",
        "status": "Ready for Discharge"
    },
    {
        "case_id": "UC-2025-0033",
        "child_initials": "F.H. (Age 15)",
        "sponsor_name": "Jorge Hernandez",
        "relationship": "Category 3 (Family Friend)",
        "days_in_care": 42, # Exceeds 30 days!
        "priority": "Urgent",
        "facility": "Tucson Residential Home",
        "status": "Ready for Discharge"
    }
]

def init_kanban_state():
    """Initializes session state storage for the Kanban case board."""
    if 'kanban_cases' not in st.session_state:
        st.session_state['kanban_cases'] = [c.copy() for c in SAMPLE_CASES]

def render_kanban_board():
    """Renders the interactive 3-column Kanban board UI."""
    init_kanban_state()
    cases = st.session_state['kanban_cases']
    
    st.markdown("### 📋 Sponsor Vetting & Workflow Kanban Board")
    st.markdown(
        "Track individual child care transitions from intake vetting through home study "
        "and final sponsor reunification. Cases exceeding **30 days in care** are flagged with priority warnings."
    )
    
    # --- TOP METRICS BAR ---
    total_active = len(cases)
    overdue_cases = [c for c in cases if c['days_in_care'] > 30]
    ready_cases = [c for c in cases if c['status'] == "Ready for Discharge"]
    avg_days = sum(c['days_in_care'] for c in cases) / total_active if total_active > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Active Cases", total_active)
    with m2:
        st.metric("Average Days in Care", f"{avg_days:.1f} Days")
    with m3:
        st.metric("Cases > 30 Days (Stagnation)", len(overdue_cases), delta=f"{len(overdue_cases)} flagged", delta_color="inverse")
    with m4:
        st.metric("Ready for Handoff", len(ready_cases), delta=f"{len(ready_cases)} ready")

    st.markdown("---")
    
    # --- FILTER CONTROLS ---
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search_query = st.text_input("🔍 Search by Case ID, Child, or Sponsor", "").lower()
    with f2:
        selected_cat = st.selectbox("Filter Category", ["All Categories", "Category 1", "Category 2", "Category 3"])
    with f3:
        selected_priority = st.selectbox("Filter Priority", ["All Priorities", "Urgent", "High", "Standard"])
        
    filtered_cases = []
    for c in cases:
        match_search = (
            search_query in c['case_id'].lower() or
            search_query in c['child_initials'].lower() or
            search_query in c['sponsor_name'].lower()
        )
        match_cat = (selected_cat == "All Categories" or selected_cat in c['relationship'])
        match_priority = (selected_priority == "All Priorities" or selected_priority == c['priority'])
        
        if match_search and match_cat and match_priority:
            filtered_cases.append(c)
            
    # --- 3-COLUMN KANBAN BOARD ---
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    col_headers = [
        ("📥 Vetting Queue", "#285A2B"),
        ("🔍 Home Study Pending", "#D97706"),
        ("✅ Ready for Discharge", "#196C2E")
    ]
    
    for idx, (col_name, header_color) in enumerate(zip(KANBAN_COLUMNS, col_headers)):
        with cols[idx]:
            col_cases = [c for c in filtered_cases if c['status'] == col_name]
            st.markdown(
                f"<div style='border-top: 4px solid {header_color[1]}; border: 1px solid #DCE7CA; border-top-width: 4px; border-top-color: {header_color[1]}; padding: 10px 14px; background: #F4F8EB; border-radius: 12px; margin-bottom: 12px; color: #173C19;'>"
                f"<b>{header_color[0]}</b> <span style='float:right; background: #FFFFFF; color: #1E4620; border: 1px solid #DCE7CA; padding: 2px 9px; border-radius: 9999px; font-size: 11px; font-weight: 700;'>{len(col_cases)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            if not col_cases:
                st.caption("No cases in this stage.")
                
            for case in col_cases:
                is_stagnant = case['days_in_care'] > 30
                border_style = "1.5px solid #DC2626" if is_stagnant else "1px solid #DCE7CA"
                aging_badge = (
                    f"<span style='background: #FDF2F2; color: #9B1C1C; border: 1px solid #F8B4B4; padding: 2px 8px; border-radius: 9999px; font-size: 10px; font-weight: 700;'>⚠️ {case['days_in_care']} DAYS (DELAY)</span>"
                    if is_stagnant else
                    f"<span style='background: #EAF7ED; color: #196C2E; border: 1px solid #C6E9CD; padding: 2px 8px; border-radius: 9999px; font-size: 10px; font-weight: 600;'>{case['days_in_care']} days</span>"
                )
                priority_color = "#9B1C1C" if case['priority'] == 'Urgent' else "#8C5B08" if case['priority'] == 'High' else "#285A2B"
                priority_bg = "#FDF2F2" if case['priority'] == 'Urgent' else "#FEF9EB" if case['priority'] == 'High' else "#F4F8EB"
                priority_badge = f"<span style='background: {priority_bg}; color: {priority_color}; padding: 2px 7px; border-radius: 9999px; font-size: 10px; font-weight: 700;'>{case['priority'].upper()}</span>"
                
                card_html = f"""
                <div style='background: #FFFFFF; border: {border_style}; border-radius: 14px; padding: 14px; margin-bottom: 10px; box-shadow: 0 4px 12px rgba(28, 56, 25, 0.04);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>
                        <b style='color: #173C19; font-size: 12px;'>{case['case_id']}</b>
                        {priority_badge}
                    </div>
                    <div style='font-size: 14px; font-weight: 700; color: #142816;'>{case['child_initials']}</div>
                    <div style='font-size: 12px; color: #485A4A; margin-top: 3px;'>Sponsor: <b style='color: #173C19;'>{case['sponsor_name']}</b></div>
                    <div style='font-size: 11px; color: #526655; margin-top: 2px;'>{case['relationship']}</div>
                    <div style='font-size: 11px; color: #718473; margin-top: 2px;'>Facility: {case['facility']}</div>
                    <div style='margin-top: 10px; display: flex; justify-content: space-between; align-items: center;'>
                        {aging_badge}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Move buttons
                b_cols = st.columns(2)
                if idx > 0:
                    with b_cols[0]:
                        if st.button("⬅️ Step Back", key=f"back_{case['case_id']}"):
                            case['status'] = KANBAN_COLUMNS[idx - 1]
                            st.rerun()
                if idx < 2:
                    with b_cols[1]:
                        if st.button("Advance ➡️", key=f"next_{case['case_id']}"):
                            case['status'] = KANBAN_COLUMNS[idx + 1]
                            st.rerun()

    # --- ADD NEW CASE EXPANDER ---
    with st.expander("➕ Intake New Case into Workflow"):
        with st.form("new_case_form"):
            c_a, c_b = st.columns(2)
            with c_a:
                new_id = st.text_input("Case ID", f"UC-2025-{len(cases)+100:04d}")
                new_child = st.text_input("Child Initial & Age", "M.D. (Age 13)")
                new_sponsor = st.text_input("Sponsor Full Name", "Ana Gomez")
            with c_b:
                new_rel = st.selectbox("Sponsor Category", [
                    "Category 1 (Parent/Legal Guardian)",
                    "Category 2 (Close Relative: Sibling/Aunt/Uncle/Grandparent)",
                    "Category 3 (Distant Relative/Other Sponsor)"
                ])
                new_priority = st.selectbox("Intake Priority", ["High", "Standard", "Urgent"])
                new_facility = st.text_input("Assigned Shelter Facility", "Southwest Care Center")
                
            submitted = st.form_submit_button("Submit Intake to Vetting Queue")
            if submitted:
                st.session_state['kanban_cases'].append({
                    "case_id": new_id,
                    "child_initials": new_child,
                    "sponsor_name": new_sponsor,
                    "relationship": new_rel,
                    "days_in_care": 1,
                    "priority": new_priority,
                    "facility": new_facility,
                    "status": "Vetting Queue"
                })
                st.success(f"Case {new_id} successfully enrolled in Vetting Queue!")
                st.rerun()
