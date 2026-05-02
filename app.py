import streamlit as st
import pandas as pd
import ollama
import os
import json
import re
import time
import shutil
import stat
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
BASE_DIR = "projects"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

st.set_page_config(page_title="AI Safety Platform", page_icon="🛡️", layout="wide")
st.title("🛡️ T&S Batch Processing & HITL Platform")

POLICY_MAP = {
    "Safe": "Content is safe and adheres to community guidelines.",
    "1.1 Identity Hate": "Content must not contain slurs, discrimination, or expressions of hatred directed at individuals or groups based on race, religion, gender, sexual orientation, disability, or demographic identity.",
    "1.2 Threats of Violence": "Content must not express intentions, plans, or desires to inflict physical pain, injury, death, or material damage against any person, group, or property.",
    "1.3 Severe Toxicity": "Content must not contain extreme, disproportionate hostility or rage, including encouraging individuals to commit self-harm or suicide.",
    "2.1 Targeted Insults": "Content must not contain demeaning, derogatory, or highly offensive remarks directed specifically at an individual with the intent to belittle or harass them.",
    "2.2 Obscenity": "Content must not contain extreme profanity, explicitly sexually vulgar language, or grotesque imagery.",
    "2.3 General Toxicity": "Content must not be generally rude, highly disrespectful, or explicitly designed to derail healthy conversation with inflammatory remarks.",
    "Unknown": "Violation policy line could not be determined."
}

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'hitl_trend_filter' not in st.session_state:
    st.session_state.hitl_trend_filter = "All"
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'cancel_processing' not in st.session_state:
    st.session_state.cancel_processing = False
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'studio_idx' not in st.session_state:
    st.session_state.studio_idx = 0
if 'esc_studio_idx' not in st.session_state:
    st.session_state.esc_studio_idx = 0

# ==========================================
# 3. DATA MANAGEMENT & FILE UTILS
# ==========================================
def force_delete_path(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def get_project_file(filename, project_name):
    if not project_name: return None
    return os.path.join(BASE_DIR, project_name, filename)

def load_data(project_name):
    file_path = get_project_file("processed_data.csv", project_name)
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        text_cols = ["violation_category", "policy_line_violated", "reviewer_notes", "trend_name", "trend_description"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        return df
    return pd.DataFrame()

def save_data(df, project_name):
    file_path = get_project_file("processed_data.csv", project_name)
    df.to_csv(file_path, index=False)

def initialize_columns(df):
    required_cols = {
        "status": None,
        "violation_category": "Unknown",
        "confidence_score": 0.0,
        "human_reviewed": False,
        "trend_name": "Uncategorized",
        "trend_description": "",
        "policy_line_violated": "",
        "reviewer_notes": ""
    }
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
    return df

# ==========================================
# 4. LLM ENGINE (CASCADE ARCHITECTURE)
# ==========================================
def process_single_row(row, system_prompt, model, temp, top_p):
    try:
        response = ollama.chat(
            model=model, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': str(row['content'])}
            ],
            options={'temperature': temp, 'top_p': top_p}
        )
        raw_text = response['message']['content']
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group(0))
            score = float(result.get("confidence_score", 0.0))
            is_violative = result.get("violates_policy", False)
            
            if score < 0.8:
                status = "ESCALATE"
                category = "Requires Human Review"
            elif is_violative:
                status = "FAIL"
                category = "Pending Deep Analysis" 
            else:
                status = "PASS"
                category = "Safe"
                
            return {"status": status, "violation_category": category, "confidence_score": score}
        else:
            return {"status": "PARSE_ERROR", "violation_category": "Unknown", "confidence_score": 0.0}
    except Exception as e:
        return {"status": "ERROR", "violation_category": "System Error", "confidence_score": 0.0, "error_msg": str(e)}

def run_trend_analysis(df, model, temp, top_p):
    status_text = st.empty()
    status_text.info("🔍 Pass 2: Running Deep Analysis & Policy Mapping on Violations...")
    
    fails_df = df[(df['status'] == 'FAIL') & (df['trend_name'] == 'Uncategorized')]
    if fails_df.empty:
        status_text.success("No new violations require trend mapping.")
        return df
    
    sample_df = fails_df.sort_values(by='confidence_score', ascending=False).head(150)
    payload = sample_df[['content']].to_dict(orient='index')
    
    prompt = f"""
    You are a Trust & Safety Senior Analyst. Analyze this JSON map of {len(sample_df)} confirmed policy violations. 
    The keys are the database IDs. The values contain the violative text.
    
    Task 1: Determine the overarching violation category (e.g., Harassment, Threat, Hate Speech, Toxicity).
    Task 2: Cluster these items into specific sub-trends. 
    Task 3: Match the violation to EXACTLY ONE of the following official policy lines:
    {list(POLICY_MAP.keys())}
    
    OUTPUT ONLY VALID JSON. Format exactly like this:
    {{
        "trends": [
            {{
                "violation_category": "Threat",
                "trend_name": "Sarcastic Political Threat",
                "trend_description": "Users masking threats against politicians using heavy sarcasm.",
                "policy_line_violated": "1.2 Threats of Violence",
                "row_indices": [12, 45, 88] 
            }}
        ]
    }}
    
    Data to analyze:
    {json.dumps(payload)}
    """
    
    try:
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': prompt}], 
            options={'temperature': 0.2, 'top_p': top_p}
        )
        raw_text = response['message']['content']
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if json_match:
            data = json.loads(json_match.group(0))
            for trend in data.get("trends", []):
                v_cat = trend.get("violation_category", "General Violation")
                t_name = trend.get("trend_name", "Uncategorized")
                t_desc = trend.get("trend_description", "")
                t_pol = trend.get("policy_line_violated", "")
                indices = trend.get("row_indices", [])
                
                for idx in indices:
                    try:
                        idx_int = int(idx)
                        if idx_int in df.index:
                            df.at[idx_int, "violation_category"] = v_cat
                            df.at[idx_int, "trend_name"] = t_name
                            df.at[idx_int, "trend_description"] = t_desc
                            df.at[idx_int, "policy_line_violated"] = t_pol
                    except ValueError:
                        pass
            status_text.success("✅ Deep Analysis & Trend Mapping complete!")
        else:
            status_text.warning("Trend mapping failed to parse JSON. Skipping.")
    except Exception as e:
        status_text.error(f"Trend mapping error: {e}")
        
    return df

def process_batch(df, thinking_mode, selected_model, temp, top_p):
    df = initialize_columns(df)
    pending_indices = df[df['status'].isna() | (df['status'] == '')].index.tolist()
    total_pending = len(pending_indices)

    if total_pending > 0:
        progress_bar = st.progress(0)
        telemetry_container = st.empty()
        start_time = time.time()
        processed_count = 0
        halt_error = None

        prompt_prefix = "<|think|>\n" if thinking_mode else ""
        system_prompt = f"""{prompt_prefix}
            You are a fast Trust & Safety Triage AI. Your ONLY job is to determine if the user content violates standard safety policies (hate speech, threats, harassment, severe toxicity).
            OUTPUT ONLY VALID JSON. Format exactly like this:
            {{
                "violates_policy": true or false,
                "confidence_score": 0.95
            }}"""

        last_ui_update = time.time() 
        
        for idx in pending_indices:
            if st.session_state.cancel_processing: break
                
            result = process_single_row(df.loc[idx], system_prompt, selected_model, temp, top_p)
            
            if result.get("status") == "ERROR":
                halt_error = f"Row {idx}: {result.get('error_msg')}"
                break
                
            df.at[idx, "status"] = result.get("status")
            df.at[idx, "violation_category"] = result.get("violation_category")
            df.at[idx, "confidence_score"] = result.get("confidence_score")
            
            processed_count += 1
            
            current_time = time.time()
            if current_time - last_ui_update > 1.0 or processed_count == total_pending:
                progress_bar.progress(processed_count / total_pending)
                elapsed = current_time - start_time
                avg_time_per_row = elapsed / processed_count if processed_count > 0 else 0
                eta_seconds = (total_pending - processed_count) * avg_time_per_row
                
                with telemetry_container.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows Processed", f"{processed_count} / {total_pending}")
                    c2.metric("Speed", f"{1/avg_time_per_row:.2f} rows/sec" if avg_time_per_row > 0 else "0.00")
                    c3.metric("ETA", f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s")
                last_ui_update = current_time 

        telemetry_container.empty()
        progress_bar.empty()
        
        if halt_error: return df, halt_error

    if not st.session_state.cancel_processing:
        df = run_trend_analysis(df, selected_model, temp, top_p)

    return df, None

# ==========================================
# 5. UI LAYOUT & SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📂 Workspace")
    new_proj = st.text_input("Create New Project:")
    if st.button("Create") and new_proj:
        os.makedirs(os.path.join(BASE_DIR, new_proj), exist_ok=True)
        st.success(f"Project '{new_proj}' created!")
        time.sleep(0.5)
        st.rerun()
        
    st.divider()
    st.subheader("Active Projects")
    projects = [f.name for f in os.scandir(BASE_DIR) if f.is_dir()]
    
    if not projects:
        st.info("No projects found.")
    else:
        for proj in projects:
            col1, col2 = st.columns([4, 1])
            with col1:
                btn_type = "primary" if st.session_state.selected_project == proj else "secondary"
                if st.button(f"📁 {proj}", key=f"select_{proj}", type=btn_type, use_container_width=True):
                    st.session_state.selected_project = proj
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{proj}", help=f"Delete {proj}"):
                    target_dir = os.path.join(BASE_DIR, proj)
                    try: shutil.rmtree(target_dir, onexc=force_delete_path)
                    except PermissionError:
                        time.sleep(0.5)
                        try: shutil.rmtree(target_dir, onexc=force_delete_path)
                        except Exception: st.error("File locked. Try again.")
                    
                    if st.session_state.selected_project == proj:
                        st.session_state.selected_project = None
                    time.sleep(0.2)
                    st.rerun()

selected_project = st.session_state.selected_project
if not selected_project:
    st.info("👈 Please create or select a project in the sidebar to begin.")
    st.stop()

st.header(f"📁 Project Active: {selected_project}")

with st.expander("⚙️ Engine Configuration", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1: selected_model = st.selectbox("Gemma 4 Model:", ["gemma4:e2b", "gemma4:e4b"], index=0)
    with col2:
        temp = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.05)
    with col3:
        st.write("**Advanced Options**")
        thinking_mode = st.checkbox("Enable 'Thinking' Mode", value=False)
        
st.divider()

tab1, tab2, tab3 = st.tabs(["📥 Batch Ingestion", "📊 Dashboard", "⚖️ HITL Review Desk"])

# --- TAB 1: BATCH INGESTION ---
with tab1:
    st.subheader("Upload CSV for Auditing")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", disabled=st.session_state.is_processing)
    
    if uploaded_file:
        try:
            preview_df = pd.read_csv(uploaded_file)
            if preview_df.empty: st.error("❌ Error: CSV is empty.")
            else:
                target_col = st.selectbox("Text Column to Audit:", options=preview_df.columns.tolist(), disabled=st.session_state.is_processing)
                
                with st.expander("👀 Preview Upload", expanded=False):
                    rows, cols = preview_df.shape
                    st.write(f"**Dataset Size:** {rows} Rows  |  {cols} Columns")
                    st.dataframe(preview_df.head(5), use_container_width=True)
                
                def start_job():
                    st.session_state.is_processing = True
                    st.session_state.cancel_processing = False
                def stop_job():
                    st.session_state.cancel_processing = True
                    st.session_state.is_processing = False

                c1, c2 = st.columns(2)
                c1.button("🚀 Start Engine" if not st.session_state.is_processing else "⚙️ Processing...", use_container_width=True, on_click=start_job, disabled=st.session_state.is_processing)
                c2.button("🛑 Cancel", use_container_width=True, on_click=stop_job, disabled=not st.session_state.is_processing)

                if st.session_state.is_processing:
                    if target_col != 'content':
                        if 'content' in preview_df.columns: preview_df = preview_df.rename(columns={'content': 'original_content'})
                        preview_df = preview_df.rename(columns={target_col: 'content'})

                    existing_df = load_data(selected_project)
                    if not existing_df.empty: df_to_process = pd.concat([existing_df, preview_df]).drop_duplicates(subset=['content'], keep='first').reset_index(drop=True)
                    else: df_to_process = preview_df
                    
                    try:
                        df_to_process, halt_error = process_batch(df_to_process, thinking_mode, selected_model, temp, top_p)
                        save_data(df_to_process, selected_project)
                        st.session_state.is_processing = False
                        
                        if halt_error: st.error(f"🛑 **HALT!** {halt_error}")
                        elif st.session_state.cancel_processing: st.warning("✋ Processing canceled.")
                        else:
                            st.success("✅ Batch processing and Trend Analysis complete!")
                            time.sleep(1)
                            st.rerun() 
                    except Exception as e:
                        st.session_state.is_processing = False
                        st.error(f"❌ Processing failed: {e}")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

df = load_data(selected_project)

# --- TAB 2: DASHBOARD ---
if not df.empty:
    with tab2:
        st.subheader("Global Risk Metrics")
        
        total_cases = len(df)
        fails = df[df['status'] == 'FAIL']
        violation_rate = (len(fails) / total_cases) * 100 if total_cases > 0 else 0
        avg_conf = df['confidence_score'].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cases", total_cases)
        m2.metric("Violations Found", len(fails))
        m3.metric("Violation Rate", f"{violation_rate:.1f}%")
        m4.metric("Avg Engine Confidence", f"{avg_conf:.2f}")
        
        st.divider()
        st.subheader("📋 Policy Violations")
        if not fails.empty:
            policy_counts = fails['policy_line_violated'].value_counts().reset_index()
            policy_counts.columns = ['Policy Line', 'Count']
            fig_policy = px.bar(policy_counts, x='Policy Line', y='Count', text='Count', color_discrete_sequence=['#EF553B'])
            fig_policy.update_layout(xaxis_title="", yaxis_title="Number of Violations")
            st.plotly_chart(fig_policy, use_container_width=True)
            
        st.divider()
        st.subheader("📈 Violation Trends Chart")
        if not fails.empty:
            pol_filter = st.selectbox("Filter Chart by Policy Line", ["All"] + list(fails['policy_line_violated'].unique()))
            filtered_fails = fails if pol_filter == "All" else fails[fails['policy_line_violated'] == pol_filter]

            if not filtered_fails.empty:
                trend_counts = filtered_fails['trend_name'].value_counts().reset_index()
                trend_counts.columns = ['Trend', 'Count']

                fig = go.Figure()
                # Left Y-Axis: Bar Chart for Counts
                fig.add_trace(go.Bar(
                    x=trend_counts['Trend'], y=trend_counts['Count'],
                    name='Case Count', yaxis='y1', marker_color='royalblue', opacity=0.7
                ))
                # Right Y-Axis: Box Plot for Confidence
                for trend in trend_counts['Trend']:
                    trend_data = filtered_fails[filtered_fails['trend_name'] == trend]
                    fig.add_trace(go.Box(
                        x=trend_data['trend_name'], y=trend_data['confidence_score'],
                        name='Confidence Spread', yaxis='y2', marker_color='darkorange', showlegend=False
                    ))

                fig.update_layout(
                    yaxis=dict(title='Case Count', side='left'),
                    yaxis2=dict(title='Confidence Score', side='right', overlaying='y', range=[0, 1.05]),
                    barmode='group', showlegend=False, hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cases match this filter.")
        else:
            st.info("No violations detected yet.")
            
        st.divider()
        st.subheader("🔍 Deep Dive: Discovered Trends")
        if not fails.empty:
            # Sort trends by highest volume
            trend_sizes = fails.groupby(['violation_category', 'trend_name']).size().reset_index(name='count')
            trend_sizes = trend_sizes.sort_values(by='count', ascending=False)
            
            for _, row in trend_sizes.iterrows():
                category = row['violation_category']
                trend_name = row['trend_name']
                t_count = row['count']
                group = fails[(fails['violation_category'] == category) & (fails['trend_name'] == trend_name)]
                
                trend_desc = group['trend_description'].iloc[0]
                pol_line = group['policy_line_violated'].iloc[0]
                t_conf = group['confidence_score'].mean()
                
                with st.expander(f"📌 {category} — {trend_name} ({t_count} cases)"):
                    st.write(f"**Description:** {trend_desc}")
                    st.write(f"**Suspected Policy Violated:** `{pol_line}`")
                    st.write(f"**Average Model Confidence:** {t_conf:.2f}")
                    
                    # Collapsible Example
                    with st.expander("👀 View Example Content"):
                        st.info(group['content'].iloc[0])
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        csv_export = group.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download Cases CSV", data=csv_export, file_name=f"{trend_name}_cases.csv", mime="text/csv", key=f"dl_{category}_{trend_name}")
                    with btn_col2:
                        if st.button(f"⚖️ Review '{trend_name}' in HITL", key=f"btn_{category}_{trend_name}"):
                            st.session_state.hitl_trend_filter = trend_name
                            st.success("Filter applied! Please click the '⚖️ HITL Review Desk' tab above.")
        else:
             st.write("No trends mapped.")

    # --- TAB 3: HITL REVIEW DESK ---
    with tab3:
        # Priority Escalation Queue
        st.subheader("🚨 Priority Escalations Queue")
        escalated_df = df[(df['status'] == 'ESCALATE') & (~df['human_reviewed'])]
        
        if not escalated_df.empty:
            st.error(f"⚠️ **{len(escalated_df)} Cases Require Immediate Intervention!** Model confidence was critically low.")
            
            # --- NEW: Escalation View Toggle ---
            esc_view_mode = st.radio("Escalation Interface:", ["🗂️ Spreadsheet View", "🔎 Studio Focus Mode"], horizontal=True, key="esc_view_toggle")
            st.write("---")
            
            if esc_view_mode == "🗂️ Spreadsheet View":
                esc_edited = st.data_editor(
                    escalated_df,
                    column_config={
                        "status": st.column_config.SelectboxColumn("Status", options=["FAIL", "PASS", "ESCALATE"], required=True),
                        "policy_line_violated": st.column_config.SelectboxColumn("Policy Line", options=list(POLICY_MAP.keys())),
                        "reviewer_notes": st.column_config.TextColumn("Reviewer Comments", max_chars=200),
                        "human_reviewed": st.column_config.CheckboxColumn("Mark as Reviewed"),
                        "content": st.column_config.TextColumn("User Content", disabled=True),
                        "confidence_score": st.column_config.NumberColumn("Confidence", disabled=True)
                    },
                    use_container_width=True, hide_index=True, key="esc_editor"
                )
                if st.button("💾 Save Priority Changes", type="primary", key="esc_save_spreadsheet"):
                    esc_edited['trend_description'] = esc_edited['policy_line_violated'].map(POLICY_MAP).fillna(esc_edited['trend_description'])
                    df.update(esc_edited)
                    save_data(df, selected_project)
                    st.success("Priority cases updated!")
                    time.sleep(1)
                    st.rerun()
                    
            elif esc_view_mode == "🔎 Studio Focus Mode":
                # Ensure index doesn't go out of bounds
                if st.session_state.esc_studio_idx >= len(escalated_df): st.session_state.esc_studio_idx = 0
                
                current_row = escalated_df.iloc[st.session_state.esc_studio_idx]
                actual_idx = current_row.name
                
                st.progress((st.session_state.esc_studio_idx + 1) / len(escalated_df), text=f"Reviewing Escalation {st.session_state.esc_studio_idx + 1} of {len(escalated_df)}")
                
                col_focus, col_action = st.columns([2, 1], gap="large")
                
                with col_focus:
                    st.markdown("#### User Content")
                    # Using red st.error for Escalations to visually distinguish from the standard queue
                    st.error(current_row['content'], icon="🚨")
                    
                    c_tag1, c_tag2 = st.columns(2)
                    c_tag1.caption(f"**Engine Flag:** {current_row['violation_category']}")
                    c_tag2.caption(f"**Engine Confidence:** {current_row['confidence_score']:.2f}")

                with col_action:
                    st.markdown("#### Reviewer Actions")
                    new_status = st.selectbox("Decision Status", ["FAIL", "PASS", "ESCALATE"], index=["FAIL", "PASS", "ESCALATE"].index(current_row['status']) if current_row['status'] in ["FAIL", "PASS", "ESCALATE"] else 0, key="esc_status_sel")
                    new_pol = st.selectbox("Policy Mapping", list(POLICY_MAP.keys()), index=list(POLICY_MAP.keys()).index(current_row['policy_line_violated']) if current_row['policy_line_violated'] in POLICY_MAP else 0, key="esc_pol_sel")
                    new_notes = st.text_area("Reviewer Notes", value=current_row['reviewer_notes'], height=100, key="esc_notes_txt")
                    
                    st.write("")
                    btn_save, btn_skip = st.columns(2)
                    
                    if btn_save.button("💾 Save & Next", type="primary", use_container_width=True, key="esc_save_next"):
                        df.at[actual_idx, 'status'] = new_status
                        df.at[actual_idx, 'policy_line_violated'] = new_pol
                        df.at[actual_idx, 'reviewer_notes'] = new_notes
                        df.at[actual_idx, 'human_reviewed'] = True
                        df.at[actual_idx, 'trend_description'] = POLICY_MAP.get(new_pol, "")
                        save_data(df, selected_project)
                        
                        st.session_state.esc_studio_idx += 1
                        st.rerun()
                        
                    if btn_skip.button("⏭️ Skip", use_container_width=True, key="esc_skip_btn"):
                        st.session_state.esc_studio_idx += 1
                        st.rerun()
        else:
            st.success("No pending escalations. The critical queue is clear!")
            
        st.divider()
        st.subheader("⚖️ Standard Review Workflow")
        
        # View Toggle
        view_mode = st.radio("Review Interface:", ["🗂️ Spreadsheet View", "🔎 Studio Focus Mode"], horizontal=True, key="std_view_toggle")
        st.write("---")

        f1, f2 = st.columns([1, 2])
        with f1:
            all_trends = ["All"] + df['trend_name'].unique().tolist()
            selected_trend = st.selectbox("Filter by Trend:", all_trends, index=all_trends.index(st.session_state.hitl_trend_filter) if st.session_state.hitl_trend_filter in all_trends else 0)
            st.session_state.hitl_trend_filter = selected_trend
        with f2:
            confidence_cap = st.slider("Max Confidence Threshold (Show items below this score):", 0.0, 1.0, 1.0, 0.05)
            
        hitl_df = df[(df['status'] != 'ESCALATE') | (df['human_reviewed'])].copy() # Filter out unreviewed escalations
        if selected_trend != "All": hitl_df = hitl_df[hitl_df['trend_name'] == selected_trend]
        hitl_df = hitl_df[hitl_df['confidence_score'] <= confidence_cap]
        
        if hitl_df.empty:
            st.success("No standard cases match the current filters.")
        else:
            if view_mode == "🗂️ Spreadsheet View":
                st.write(f"**Showing {len(hitl_df)} cases for review.**")
                edited_df = st.data_editor(
                    hitl_df,
                    column_config={
                        "status": st.column_config.SelectboxColumn("Status", options=["FAIL", "PASS", "ESCALATE"], required=True),
                        "policy_line_violated": st.column_config.SelectboxColumn("Policy Line", options=list(POLICY_MAP.keys())),
                        "reviewer_notes": st.column_config.TextColumn("Reviewer Comments", max_chars=200),
                        "trend_description": st.column_config.TextColumn("Trend Description"),
                        "human_reviewed": st.column_config.CheckboxColumn("Mark as Reviewed"),
                        "content": st.column_config.TextColumn("User Content", disabled=True),
                        "confidence_score": st.column_config.NumberColumn("Confidence", disabled=True)
                    },
                    use_container_width=True, hide_index=True, key="std_editor"
                )
                if st.button("💾 Save All HITL Changes", type="primary", key="std_save_spreadsheet"):
                    edited_df['trend_description'] = edited_df['policy_line_violated'].map(POLICY_MAP).fillna(edited_df['trend_description'])
                    df.update(edited_df)
                    save_data(df, selected_project)
                    st.success("Changes saved successfully!")
                    time.sleep(1)
                    st.rerun()
            
            elif view_mode == "🔎 Studio Focus Mode":
                # Ensure index doesn't go out of bounds if filters change
                if st.session_state.studio_idx >= len(hitl_df): st.session_state.studio_idx = 0
                
                current_row = hitl_df.iloc[st.session_state.studio_idx]
                actual_idx = current_row.name # We need the original index to update the main df
                
                st.progress((st.session_state.studio_idx + 1) / len(hitl_df), text=f"Reviewing Case {st.session_state.studio_idx + 1} of {len(hitl_df)}")
                
                col_focus, col_action = st.columns([2, 1], gap="large")
                
                with col_focus:
                    st.markdown("#### User Content")
                    # Using a blue st.info for standard items
                    st.info(current_row['content'], icon="💬")
                    
                    c_tag1, c_tag2 = st.columns(2)
                    c_tag1.caption(f"**Engine Flag:** {current_row['violation_category']}")
                    c_tag2.caption(f"**Engine Confidence:** {current_row['confidence_score']:.2f}")

                with col_action:
                    st.markdown("#### Reviewer Actions")
                    new_status = st.selectbox("Decision Status", ["FAIL", "PASS", "ESCALATE"], index=["FAIL", "PASS", "ESCALATE"].index(current_row['status']) if current_row['status'] in ["FAIL", "PASS", "ESCALATE"] else 0, key="std_status_sel")
                    new_pol = st.selectbox("Policy Mapping", list(POLICY_MAP.keys()), index=list(POLICY_MAP.keys()).index(current_row['policy_line_violated']) if current_row['policy_line_violated'] in POLICY_MAP else 0, key="std_pol_sel")
                    new_notes = st.text_area("Reviewer Notes", value=current_row['reviewer_notes'], height=100, key="std_notes_txt")
                    
                    st.write("")
                    btn_save, btn_skip = st.columns(2)
                    
                    if btn_save.button("💾 Save & Next", type="primary", use_container_width=True, key="std_save_next"):
                        # Commit changes to main DataFrame
                        df.at[actual_idx, 'status'] = new_status
                        df.at[actual_idx, 'policy_line_violated'] = new_pol
                        df.at[actual_idx, 'reviewer_notes'] = new_notes
                        df.at[actual_idx, 'human_reviewed'] = True
                        df.at[actual_idx, 'trend_description'] = POLICY_MAP.get(new_pol, "")
                        save_data(df, selected_project)
                        
                        st.session_state.studio_idx += 1
                        st.rerun()
                        
                    if btn_skip.button("⏭️ Skip", use_container_width=True, key="std_skip_btn"):
                        st.session_state.studio_idx += 1
                        st.rerun()