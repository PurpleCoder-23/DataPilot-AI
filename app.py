
import hashlib
import io
import re
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_analysis import (
    load_dataset,
    get_dataset_info,
    profile_dataset,
    numerical_summary,
    categorical_summary,
    correlation_matrix,
    outlier_summary,
    remove_duplicates,
    handle_missing_values,
    drop_columns,
    rename_columns,
    convert_column_types,
    DataAnalysisError,
)




# =========================================================
# IN-MEMORY ETL FOR THE ACTIVE STREAMLIT DATASET
# =========================================================

def is_uploaded_dataset_etl_request(question: str) -> bool:
    """Return True for file transformations on the currently uploaded dataset."""
    text = (question or "").lower()
    action_terms = (
        "remove duplicate", "remove duplicates", "drop duplicate",
        "drop duplicates", "deduplicate", "sort by", "sort the",
        "convert", "export", "download", "save as", "transform",
        "clean my", "clean the", "filter rows", "drop column",
        "select columns",
    )
    return any(term in text for term in action_terms)


def transform_uploaded_dataframe(df: pd.DataFrame, question: str):
    """
    Apply safe, deterministic transformations directly to Streamlit's active
    DataFrame. This avoids asking users for a project path for an uploaded file.
    Returns (transformed_dataframe, description).
    """
    result = df.copy()
    text = (question or "").lower()
    operations = []

    # Remove duplicates when requested.
    if any(term in text for term in (
        "remove duplicate", "remove duplicates", "drop duplicate",
        "drop duplicates", "deduplicate",
    )):
        before = len(result)
        result = result.drop_duplicates().reset_index(drop=True)
        operations.append(f"Removed {before - len(result):,} duplicate row(s)")

    # Sort using an actual column name. Prefer explicitly named columns.
    sort_column = None
    for column in result.columns:
        if str(column).strip().lower() in text:
            if any(term in text for term in ("sort", "order", "ascending", "descending")):
                sort_column = column
                break

    if sort_column is None and any(term in text for term in ("sort", "order")):
        # Common natural-language request: sort by Name.
        for column in result.columns:
            if str(column).strip().lower() == "name" and "name" in text:
                sort_column = column
                break

    if sort_column is not None:
        ascending = not any(
            term in text for term in ("descending", "desc", "highest to lowest", "z to a")
        )
        result = result.sort_values(
            by=sort_column,
            ascending=ascending,
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
        direction = "ascending" if ascending else "descending"
        operations.append(f"Sorted by **{sort_column}** ({direction})")

    # Drop columns only when the user clearly requests it and the column exists.
    if "drop column" in text or "remove column" in text:
        requested_columns = [
            col for col in result.columns
            if str(col).strip().lower() in text
        ]
        if requested_columns:
            result = result.drop(columns=requested_columns)
            operations.append("Dropped column(s): " + ", ".join(map(str, requested_columns)))

    if not operations:
        operations.append("No row/column transformation was requested; data preserved")

    description = "; ".join(operations)
    return result, description


def etl_download_filename(dataset_name: str, question: str) -> str:
    """Choose a safe download filename and requested output format."""
    text = (question or "").lower()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(dataset_name or "dataset"))
    stem = re.sub(r"_+", "_", stem).strip("_") or "dataset"

    # This chat's in-memory ETL supports CSV and Excel downloads.
    if "excel" in text or ".xlsx" in text or "xlsx" in text:
        return f"{stem}_transformed.xlsx"
    return f"{stem}_transformed.csv"





def generate_sql_for_uploaded_dataset(question: str, df: pd.DataFrame):
    """
    Generate simple SELECT SQL from the uploaded DataFrame schema.
    This is SQL text only; it is not executed against SQL Server.
    Returns (sql, explanation) or (None, None) if the request is not supported.
    """
    q = (question or "").strip().lower()
    if not any(term in q for term in ("sql", "query", "select")):
        return None, None

    # Resolve requested sorting column from actual uploaded columns.
    sort_col = None
    for col in df.columns:
        col_text = str(col).strip().lower()
        if col_text and col_text in q and any(
            word in q for word in ("sort", "order", "ascending", "descending", "order by")
        ):
            sort_col = col
            break

    # SQL generation for sort/order-by requests.
    if sort_col is not None:
        direction = "DESC" if any(
            term in q for term in ("descending", "desc", "highest to lowest", "z to a")
        ) else "ASC"
        sql = (
            "SELECT *\n"
            "FROM [YourTable]\n"
            f"ORDER BY [{str(sort_col).replace(']', ']]')} ] {direction};"
        )
        # Avoid a stray space before closing bracket in the generated identifier.
        sql = sql.replace("] ]", "]]")
        return sql, f"Sorts rows by [{sort_col}] in {direction.lower()} order."

    # Count by a column when explicitly requested.
    group_col = None
    if any(term in q for term in ("count", "number of", "how many")) and any(
        term in q for term in ("each", "per", "group by", "grouped")
    ):
        for col in df.columns:
            col_text = str(col).strip().lower()
            if col_text and col_text in q:
                group_col = col
                break

    if group_col is not None:
        escaped = str(group_col).replace("]", "]]")
        sql = (
            f"SELECT [{escaped}], COUNT(*) AS [RecordCount]\n"
            "FROM [YourTable]\n"
            f"GROUP BY [{escaped}]\n"
            "ORDER BY [RecordCount] DESC;"
        )
        return sql, f"Counts records by [{group_col}] in descending count order."

    # Generic all rows request, only if explicitly asking for all rows/columns.
    if any(term in q for term in ("all rows", "all records", "select *", "entire table")):
        return "SELECT *\nFROM [YourTable];", "Returns all rows and columns."

    return None, None



# =========================================================
# PAGE CONFIG
# =========================================================

from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="DATA AGENT | Intelligence Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b0e14;
        color: #e8eaf0;
    }

    [data-testid="stSidebar"] {
        background: #10141d;
        border-right: 1px solid #242a38;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #f3f4f8 !important;
        letter-spacing: -0.5px;
    }

    .agent-brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #ffffff;
    }

    .agent-brand span {
        color: #a78bfa;
    }

    .agent-subtitle {
        color: #8993a7;
        font-size: 12px;
        margin-top: 3px;
    }

    .workspace-header {
        background: linear-gradient(115deg, #171b2b, #111521);
        border: 1px solid #292f43;
        border-radius: 18px;
        padding: 28px 30px;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #a78bfa;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .workspace-title {
        font-size: 30px;
        font-weight: 800;
        color: #f4f4f8;
        margin-top: 8px;
    }

    .workspace-description {
        color: #9aa3b5;
        font-size: 14px;
        margin-top: 8px;
    }

    .status-pill {
        display: inline-block;
        background: #172c29;
        color: #5ee6b2;
        border: 1px solid #285247;
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 600;
    }

    .metric-card {
        background: #141925;
        border: 1px solid #272e3d;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 115px;
    }

    .metric-label {
        color: #8e99ad;
        font-size: 12px;
        font-weight: 600;
    }

    .metric-value {
        color: #f3f4f8;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #727f95;
        font-size: 11px;
        margin-top: 4px;
    }

    .section-label {
        color: #a78bfa;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .agent-card {
        background: #141925;
        border: 1px solid #292f40;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 12px;
    }

    .agent-card-title {
        color: #f1f3f8;
        font-weight: 700;
        font-size: 16px;
    }

    .agent-card-description {
        color: #8994a8;
        font-size: 13px;
        margin-top: 8px;
    }

    .agent-icon {
        font-size: 25px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: #141925;
        border: 1px solid #272e3d;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa3b5;
    }

    div[data-testid="stMetricValue"] {
        color: #f3f4f8;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #252c3a;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8e99ad;
        border-radius: 8px 8px 0 0;
        padding: 12px 18px;
    }

    .stTabs [aria-selected="true"] {
        color: #c4b5fd !important;
        border-bottom: 2px solid #8b5cf6 !important;
    }

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #3a315c;
        background: #211b36;
        color: #d8ccff;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #332650;
        border-color: #8b5cf6;
        color: white;
    }

    .stDownloadButton > button {
        background: #201b35;
        color: #d8ccff;
        border: 1px solid #514075;
        border-radius: 9px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #292f40;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #252c3a;
    }

    /* Chat input: high-contrast typing and placeholder */
    [data-testid="stChatInput"] {
        background: #f8fafc !important;
        border: 1px solid #8b5cf6 !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #6d28d9 !important;
        background: #f8fafc !important;
        opacity: 1 !important;
        font-size: 16px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #4b5563 !important;
        -webkit-text-fill-color: #4b5563 !important;
        opacity: 1 !important;
    }

    /* Make chat messages readable on the dark workspace */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #f1f5f9 !important;
    }

    /* Sidebar navigation buttons */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        min-height: 44px;
        background: #151b28;
        border: 1px solid #283247;
        color: #e5e7eb;
        margin: 3px 0;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #29213f;
        border-color: #8b5cf6;
        color: #ffffff;
    }

    /* Prominent chatbot button */
    [data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(90deg, #6d28d9, #8b5cf6) !important;
        color: #ffffff !important;
        border: 1px solid #a78bfa !important;
        font-weight: 700 !important;
    }


    /* Main chatbot workspace redesign */
    .chat-hero {
        padding: 28px 30px 22px 30px;
        margin: 0 0 22px 0;
        border-radius: 22px;
        background: linear-gradient(120deg, #17152d 0%, #211a42 55%, #17253b 100%);
        border: 1px solid rgba(167, 139, 250, 0.35);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
    }

    .chat-hero-top {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .chat-logo {
        width: 64px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 18px;
        font-size: 30px;
        background: linear-gradient(135deg, #7c3aed, #a78bfa);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3);
    }

    .chat-eyebrow {
        color: #c4b5fd;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .chat-hero h1 {
        color: #ffffff !important;
        font-size: 34px !important;
        line-height: 1.15 !important;
        margin: 0 !important;
        font-weight: 750 !important;
    }

    .chat-hero p {
        color: #d6d3e8 !important;
        margin: 6px 0 0 0 !important;
        font-size: 15px !important;
    }

    .chat-hero-bottom {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 22px;
    }

    .hero-chip {
        border: 1px solid rgba(196, 181, 253, 0.28);
        background: rgba(255, 255, 255, 0.06);
        color: #e9e5ff;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
    }

    .dataset-status {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 15px 18px;
        border-radius: 15px;
        margin: 0 0 24px 0;
        border: 1px solid rgba(52, 211, 153, 0.3);
        background: rgba(16, 185, 129, 0.08);
    }

    .dataset-empty {
        border-color: rgba(167, 139, 250, 0.3);
        background: rgba(139, 92, 246, 0.08);
    }

    .dataset-status-icon {
        font-size: 24px;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.07);
    }

    .dataset-status-text {
        display: flex;
        flex-direction: column;
        flex: 1;
        gap: 3px;
    }

    .dataset-status-text strong {
        color: #f8fafc;
        font-size: 14px;
    }

    .dataset-status-text span {
        color: #b8c1d1;
        font-size: 12px;
        overflow-wrap: anywhere;
    }

    .dataset-status-pill {
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 700;
    }

    .dataset-empty .dataset-status-pill {
        color: #c4b5fd;
        background: rgba(139, 92, 246, 0.12);
        border-color: rgba(167, 139, 250, 0.25);
    }

    .welcome-block {
        margin: 26px 0 12px 0;
    }

    .welcome-block h2 {
        color: #f8fafc !important;
        font-size: 24px !important;
        margin-bottom: 5px !important;
    }

    .welcome-block p {
        color: #aab4c5 !important;
        font-size: 14px !important;
    }

    /* Prompt suggestion buttons */
    div[data-testid="stMain"] div.stButton > button {
        min-height: 58px;
        border-radius: 13px;
        text-align: left;
        padding: 12px 16px;
        background: #141b29;
        border: 1px solid #2c3548;
        color: #edf2f7;
        transition: all 0.15s ease-in-out;
    }

    div[data-testid="stMain"] div.stButton > button:hover {
        border-color: #8b5cf6;
        background: #201a36;
        color: #ffffff;
    }

    /* Main chat input stays readable while typing */
    [data-testid="stChatInput"] {
        background: #f8fafc !important;
        border: 1px solid #8b5cf6 !important;
        border-radius: 14px !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.08) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #6d28d9 !important;
        background: #f8fafc !important;
        font-size: 16px !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #4b5563 !important;
        -webkit-text-fill-color: #4b5563 !important;
        opacity: 1 !important;
    }

    /* Streamlit chat message typography */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #f1f5f9 !important;
        line-height: 1.65;
    }


    .chat-upload-heading {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 18px 0 8px 0;
        color: #f8fafc;
    }

    .chat-upload-heading > span {
        font-size: 24px;
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
    }

    .chat-upload-heading strong {
        display: block;
        font-size: 14px;
        color: #f8fafc;
    }

    .chat-upload-heading small {
        display: block;
        margin-top: 3px;
        color: #aab4c5;
        font-size: 12px;
    }

    /* Upload widget in the main chat area */
    div[data-testid="stMain"] [data-testid="stFileUploader"] {
        border: 1px dashed #6d5ca8;
        background: rgba(139, 92, 246, 0.06);
        border-radius: 14px;
        padding: 10px 14px;
        margin-bottom: 16px;
    }

    div[data-testid="stMain"] [data-testid="stFileUploader"] section {
        background: transparent;
    }


    /* High contrast for SQL and other Streamlit code blocks */
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code {
        background-color: #111827 !important;
        color: #f9fafb !important;
        border-color: #374151 !important;
    }

    [data-testid="stCodeBlock"] code span {
        color: #f9fafb !important;
    }

    /* Markdown fenced code blocks rendered by st.markdown (SQL response) */
    [data-testid="stMarkdown"] pre,
    [data-testid="stMarkdown"] pre code,
    .stMarkdown pre,
    .stMarkdown pre code,
    .stMarkdown code,
    div[data-testid="stChatMessage"] pre,
    div[data-testid="stChatMessage"] pre code {
        background: #111827 !important;
        color: #f9fafb !important;
        -webkit-text-fill-color: #f9fafb !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
    }

    [data-testid="stMarkdown"] pre,
    .stMarkdown pre,
    div[data-testid="stChatMessage"] pre {
        padding: 16px 18px !important;
        overflow-x: auto !important;
    }

    [data-testid="stMarkdown"] pre code *,
    .stMarkdown pre code *,
    div[data-testid="stChatMessage"] pre code * {
        color: #f9fafb !important;
        -webkit-text-fill-color: #f9fafb !important;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "dataset_upload_hash" not in st.session_state:
    st.session_state.dataset_upload_hash = None

if "original_dataset" not in st.session_state:
    st.session_state.original_dataset = None

if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []

if "etl_result_df" not in st.session_state:
    st.session_state.etl_result_df = None

if "etl_result_filename" not in st.session_state:
    st.session_state.etl_result_filename = None

if "etl_result_message" not in st.session_state:
    st.session_state.etl_result_message = None



# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="agent-brand">◈ DATA <span>AGENT</span></div>
    <div class="agent-subtitle">INTELLIGENCE WORKSPACE</div>
    """, unsafe_allow_html=True)

    st.divider()

    # One-time homepage migration: force Agent Chat on first load of this
    # corrected version, then preserve later sidebar navigation selections.
    if "chat_home_migration_v3" not in st.session_state:
        st.session_state.active_page = "Agent Chat"
        st.session_state.chat_home_migration_v3 = True

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Agent Chat"

    # A dedicated robot button makes Agent Chat easy to find.
    if st.button(
        "🤖  Agent Chat",
        key="sidebar_agent_chat",
        type="primary",
        help="Open the AI chatbot for your uploaded dataset",
    ):
        st.session_state.active_page = "Agent Chat"
        st.rerun()

    st.markdown("#### Workspace")

    navigation_items = [
        ("▦", "Overview"),
        ("◫", "Profiling"),
        ("✦", "Data Cleaning"),
        ("▤", "Visualization"),
        ("⇩", "Export"),
    ]

    for nav_icon, nav_label in navigation_items:
        button_label = f"{nav_icon}  {nav_label}"
        if st.button(
            button_label,
            key=f"nav_{nav_label.lower().replace(' ', '_')}",
        ):
            st.session_state.active_page = nav_label
            st.rerun()

    page = st.session_state.active_page

    st.divider()

    st.markdown("#### Data Source")

    # Dataset upload is intentionally located in the main chat workspace.

    if st.session_state.dataset is not None:

        st.markdown(
            f"**Active dataset**  \n"
            f"`{st.session_state.dataset_name}`"
        )

        if st.button("Reset to Original Dataset"):

            st.session_state.dataset = (
                st.session_state.original_dataset.copy()
            )

            st.rerun()

        if st.button("Remove Dataset"):

            st.session_state.dataset = None
            st.session_state.original_dataset = None
            st.session_state.dataset_name = None
            st.session_state.dataset_upload_hash = None
            st.session_state.etl_result_df = None
            st.session_state.etl_result_filename = None
            st.session_state.etl_result_message = None

            st.rerun()

    st.divider()

    st.markdown("""
    <div style="color:#788398;font-size:11px;">
        DATA AGENT v1.0<br>
        AI DATA INTELLIGENCE
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown("""
<div class="workspace-header">
    <div class="eyebrow">AI DATA INTELLIGENCE PLATFORM</div>
    <div class="workspace-title">Your Data. Your Intelligence.</div>
    <div class="workspace-description">
        Analyze datasets, uncover patterns, clean data,
        and transform information into insights.
    </div>
    <br>
    <span class="status-pill">● Workspace Online</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# ACTIVE DATASET
# =========================================================

df = st.session_state.dataset
info = get_dataset_info(df) if df is not None else None


# =========================================================
# MAIN CHAT WORKSPACE
# =========================================================

if page == "Agent Chat":

    # Main hero / product identity
    st.markdown(
        """
        <div class="chat-hero">
            <div class="chat-hero-top">
                <div class="chat-logo">🤖</div>
                <div>
                    <div class="chat-eyebrow">YOUR INTELLIGENT WORKSPACE</div>
                    <h1>Data Agent</h1>
                    <p>Ask questions. Explore data. Get insights.</p>
                </div>
            </div>
            <div class="chat-hero-bottom">
                <span class="hero-chip">✦ Data Analyst</span>
                <span class="hero-chip">▤ SQL Analyst</span>
                <span class="hero-chip">⇄ ETL Analyst</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset attachment area inside the main chat workspace.
    st.markdown(
        """
        <div class="chat-upload-heading">
            <span>📎</span>
            <div>
                <strong>Attach a dataset</strong>
                <small>Upload CSV, Excel, or JSON directly in this chat workspace.</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=["csv", "xlsx", "xls", "json"],
        help="Supported files up to 50 MB",
        key="chat_dataset_uploader",
        label_visibility="collapsed",
    )

    if chat_uploaded_file is not None:
        # Compare file content, not just its name. This allows replacing a file
        # with a different dataset even when both files have the same filename.
        uploaded_bytes = chat_uploaded_file.getvalue()
        upload_hash = hashlib.sha256(uploaded_bytes).hexdigest()

        if upload_hash != st.session_state.dataset_upload_hash:
            try:
                uploaded_df = load_dataset(chat_uploaded_file)

                st.session_state.dataset = uploaded_df.copy()
                st.session_state.original_dataset = uploaded_df.copy()
                st.session_state.dataset_name = chat_uploaded_file.name
                st.session_state.dataset_upload_hash = upload_hash
                st.session_state.etl_result_df = None
                st.session_state.etl_result_filename = None
                st.session_state.etl_result_message = None

                st.success(
                    f"Loaded **{chat_uploaded_file.name}** "
                    f"({len(uploaded_df):,} rows, {len(uploaded_df.columns):,} columns)"
                )
                st.rerun()

            except DataAnalysisError as error:
                st.error(str(error))

            except Exception as error:
                st.error(f"Upload failed: {type(error).__name__}: {error}")


    # Dataset status panel
    if st.session_state.dataset is not None:
        dataset_rows, dataset_cols = st.session_state.dataset.shape
        st.markdown(
            f"""
            <div class="dataset-status">
                <div class="dataset-status-icon">📊</div>
                <div class="dataset-status-text">
                    <strong>Dataset connected</strong>
                    <span>{st.session_state.dataset_name} ·
                    {dataset_rows:,} rows · {dataset_cols:,} columns</span>
                </div>
                <div class="dataset-status-pill">Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="dataset-status dataset-empty">
                <div class="dataset-status-icon">📂</div>
                <div class="dataset-status-text">
                    <strong>No dataset connected yet</strong>
                    <span>Attach a CSV, Excel, or JSON file above to begin.</span>
                </div>
                <div class="dataset-status-pill">Waiting</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Welcome state / suggested prompts when there is no conversation yet
    if not st.session_state.agent_chat_history:
        st.markdown(
            """
            <div class="welcome-block">
                <h2>What would you like to explore?</h2>
                <p>Start with a question or choose one of these suggestions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prompt_cols = st.columns(2, gap="medium")
        starter_prompts = [
            ("📈", "Explore my dataset", "Give me an overview of my uploaded dataset"),
            ("🧹", "Find data quality issues", "Check missing values and duplicate rows"),
            ("🔎", "Understand a column", "Give me statistics for a column in my dataset"),
            ("🗄️", "Ask SQL questions", "Show me what SQL database questions you can answer"),
        ]

        for index, (emoji, title, prompt_text) in enumerate(starter_prompts):
            with prompt_cols[index % 2]:
                if st.button(
                    f"{emoji}  {title}",
                    key=f"starter_prompt_{index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_agent_prompt = prompt_text
                    st.rerun()

    # Conversation history
    for chat_message in st.session_state.agent_chat_history:
        with st.chat_message(chat_message["role"]):
            st.markdown(chat_message["content"])

    # Chat input
    pending_prompt = st.session_state.pop("pending_agent_prompt", None)
    user_prompt = st.chat_input(
        "Ask your Data Agent anything...",
        key="main_agent_chat_input",
    )

    if pending_prompt:
        user_prompt = pending_prompt

    if user_prompt:
        st.session_state.agent_chat_history.append(
            {"role": "user", "content": user_prompt}
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking and analyzing..."):
                try:
                    # Uploaded-file ETL is handled in memory by Streamlit.
                    # Streamlit UploadedFile objects are temporary and the
                    # DataFrame is already available in session_state, so no
                    # project-relative path should be requested for this case.
                    uploaded_sql, uploaded_sql_explanation = (
                        generate_sql_for_uploaded_dataset(
                            user_prompt,
                            st.session_state.dataset,
                        )
                        if st.session_state.dataset is not None
                        else (None, None)
                    )

                    if uploaded_sql is not None:
                        sql_response = (
                            "### Generated SQL query (not executed):\\n\\n"
                            "```sql\\n"
                            f"{uploaded_sql}\\n"
                            "```\\n\\n"
                            f"{uploaded_sql_explanation}\\n\\n"
                            "**Note:** Replace `[YourTable]` with the actual SQL table "
                            "name. This query uses the columns from your uploaded "
                            "dataset and has not been executed."
                        )
                        result = {
                            "messages": [AIMessage(content=sql_response)],
                            "route_response": "data_analysis",
                        }
                    elif (
                        st.session_state.dataset is not None
                        and is_uploaded_dataset_etl_request(user_prompt)
                    ):
                        transformed_df, transformation_summary = (
                            transform_uploaded_dataframe(
                                st.session_state.dataset,
                                user_prompt,
                            )
                        )
                        output_filename = etl_download_filename(
                            st.session_state.dataset_name,
                            user_prompt,
                        )
                        st.session_state.etl_result_df = transformed_df.copy()
                        st.session_state.etl_result_filename = output_filename
                        st.session_state.etl_result_message = (
                            f"### Uploaded dataset transformed successfully\\n\\n"
                            f"**Source:** `{st.session_state.dataset_name}`\\n\\n"
                            f"**Result:** {transformation_summary}\\n\\n"
                            f"**Output:** {len(transformed_df):,} rows × "
                            f"{len(transformed_df.columns):,} columns\\n\\n"
                            "Use the **Download transformed dataset** section "
                            "below the chat to save the result. The active "
                            "original dataset has not been overwritten."
                        )
                        result = {
                            "messages": [AIMessage(content=st.session_state.etl_result_message)],
                            "route_response": "etl",
                        }
                    else:
                        result = data_agent.invoke(
                            {
                                "messages": [
                                    HumanMessage(content=user_prompt)
                                ]
                            },
                            config={
                                "recursion_limit": 20,
                                "configurable": {
                                    "dataset": st.session_state.dataset,
                                    "dataset_name": st.session_state.dataset_name,
                                },
                            },
                        )

                    final_message = next(
                        (
                            message.content
                            for message in reversed(result.get("messages", []))
                            if isinstance(message, AIMessage)
                            and isinstance(message.content, str)
                            and message.content.strip()
                        ),
                        "The Data Agent did not return a response.",
                    )

                    # Some model/tool responses contain escaped newline sequences
                    # instead of real line breaks. Normalize them before Markdown
                    # rendering so SQL and lists display correctly.
                    final_message = (
                        final_message
                        .replace("\\r\\n", "\\n")
                        .replace("\\n", "\n")
                        .replace("\\t", "    ")
                    )

                    st.markdown(final_message)

                    route_name = {
                        "data_analysis": "Data Analyst · Uploaded Dataset",
                        "sql": "SQL Analyst · SQL Server",
                        "etl": "ETL Analyst · Data Extraction & Transformation",
                        "general": "Data Agent · General Conversation",
                    }.get(
                        result.get("route_response"),
                        "Data Agent",
                    )

                    st.caption(f"🤖 Specialist: **{route_name}**")

                    st.session_state.agent_chat_history.append(
                        {
                            "role": "assistant",
                            "content": final_message,
                        }
                    )

                except Exception as error:
                    error_message = (
                        f"Agent request failed: {type(error).__name__}: {error}"
                    )
                    st.error(error_message)
                    st.session_state.agent_chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )

    # Quick export of the current working dataset directly from the chat page.
    if st.session_state.dataset is not None:
        with st.expander("⬇️ Download current dataset", expanded=False):
            current_df = st.session_state.dataset
            st.caption(
                f"Export the current working version of "
                f"{st.session_state.dataset_name or 'your dataset'}."
            )
            export_csv = current_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=export_csv,
                file_name="data_agent_current_dataset.csv",
                mime="text/csv",
                key="chat_download_csv",
            )

            try:
                import io
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    current_df.to_excel(
                        writer, index=False, sheet_name="Current Dataset"
                    )
                st.download_button(
                    "Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="data_agent_current_dataset.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="chat_download_excel",
                )
            except ImportError:
                st.info("Excel export requires openpyxl. CSV export is available.")

    # ETL result downloads (separate from the original active dataset).
    if st.session_state.etl_result_df is not None:
        with st.expander("⬇️ Download transformed dataset", expanded=True):
            transformed_df = st.session_state.etl_result_df
            st.caption(
                "This is the ETL output. Your original active dataset remains unchanged."
            )
            st.dataframe(transformed_df.head(10), width="stretch")

            csv_bytes = transformed_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download transformed CSV",
                data=csv_bytes,
                file_name=(
                    st.session_state.etl_result_filename
                    if str(st.session_state.etl_result_filename).lower().endswith(".csv")
                    else re.sub(r"\.xlsx$", ".csv", st.session_state.etl_result_filename or "transformed.csv")
                ),
                mime="text/csv",
                key="etl_result_download_csv",
            )

            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    transformed_df.to_excel(
                        writer, index=False, sheet_name="Transformed Data"
                    )
                st.download_button(
                    "Download transformed Excel",
                    data=excel_buffer.getvalue(),
                    file_name=re.sub(
                        r"\.csv$", ".xlsx",
                        st.session_state.etl_result_filename or "transformed.csv",
                    ),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="etl_result_download_excel",
                )
            except ImportError:
                st.info("Excel export requires openpyxl. CSV download is available.")

    # Chat controls
    control_col1, control_col2 = st.columns([1, 5])
    with control_col1:
        if st.button("🧹 Clear chat", key="main_clear_chat"):
            st.session_state.agent_chat_history = []
            st.rerun()

    with control_col2:
        st.caption(
            "Your Main Data Agent automatically chooses the right specialist "
            "for each request."
        )

    st.stop()


# Secondary workspace pages require a dataset. Agent Chat does not.
if page != "Agent Chat" and st.session_state.dataset is None:
    st.info("Please open Agent Chat and attach a dataset to use this workspace page.")
    st.stop()


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown('<div class="section-label">Workspace Overview</div>',
                unsafe_allow_html=True)

    st.header("Dataset Intelligence")

    st.caption(
        f"Active dataset: {st.session_state.dataset_name}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{info['rows']:,}")
    col2.metric("Features", info["columns"])
    col3.metric("Missing Values", f"{info['missing_values']:,}")
    col4.metric("Duplicate Records", f"{info['duplicate_rows']:,}")

    st.divider()

    left, right = st.columns([1.5, 1])

    with left:

        st.subheader("Dataset Preview")

        preview_rows = st.slider(
            "Preview records",
            min_value=1,
            max_value=min(100, len(df)),
            value=min(10, len(df)),
        )

        st.dataframe(
            df.head(preview_rows),
            width="stretch",
        )

    with right:

        st.subheader("Dataset Health")

        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        total_cells = max(df.size, 1)

        completeness = (
            1 - missing_count / total_cells
        ) * 100

        st.metric(
            "Data Completeness",
            f"{completeness:.1f}%"
        )

        st.progress(completeness / 100)

        st.metric(
            "Memory Usage",
            f"{info['memory_usage_mb']} MB"
        )

        st.metric(
            "Unique Records",
            f"{len(df) - duplicate_count:,}"
        )

    st.divider()

    st.subheader("Available Intelligence Modules")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="agent-card">
            <div class="agent-card-title">01 / Profiling</div>
            <div class="agent-card-description">
                Missing-value analysis, distributions,
                statistical summaries, and correlations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="agent-card">
            <div class="agent-card-title">02 / Cleaning</div>
            <div class="agent-card-description">
                Remove duplicates, handle missing values,
                rename columns, and convert data types.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="agent-card">
            <div class="agent-card-title">03 / Visualization</div>
            <div class="agent-card-description">
                Explore data using interactive
                charts and visual analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PROFILING PAGE
# =========================================================

elif page == "Profiling":

    st.markdown('<div class="section-label">Data Intelligence</div>',
                unsafe_allow_html=True)

    st.header("Dataset Profiling")

    profile = profile_dataset(df)

    st.subheader("Column Quality Report")

    st.dataframe(
        profile["columns"],
        width="stretch",
    )

    st.divider()

    st.subheader("Numerical Statistics")

    numerical = numerical_summary(df)

    if numerical.empty:
        st.info("No numerical columns found.")
    else:
        st.dataframe(numerical, width="stretch")

    st.divider()

    st.subheader("Categorical Analysis")

    categorical = categorical_summary(df)

    if not categorical:

        st.info("No categorical columns found.")

    else:

        selected_category = st.selectbox(
            "Select categorical feature",
            list(categorical.keys()),
        )

        st.dataframe(
            categorical[selected_category],
            width="stretch",
        )

    st.divider()

    st.subheader("Correlation Analysis")

    correlations = correlation_matrix(df)

    if correlations.empty:

        st.info(
            "At least two numerical columns are required "
            "for correlation analysis."
        )

    else:

        fig = px.imshow(
            correlations,
            text_auto=True,
            color_continuous_scale="Purples",
            title="Feature Correlation Matrix",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    st.divider()

    st.subheader("Outlier Intelligence")

    outliers = outlier_summary(df)

    if outliers.empty:
        st.info("No numerical columns available.")
    else:
        st.dataframe(outliers, width="stretch")


# =========================================================
# DATA CLEANING PAGE
# =========================================================

elif page == "Data Cleaning":

    st.markdown('<div class="section-label">Data Preparation</div>',
                unsafe_allow_html=True)

    st.header("Data Cleaning Workspace")

    st.warning(
        "Cleaning operations modify your working dataset. "
        "Use Reset to Original Dataset in the sidebar "
        "to restore the uploaded data."
    )

    st.subheader("Remove Duplicate Records")

    keep_option = st.selectbox(
        "Duplicate handling",
        [
            "Keep first occurrence",
            "Keep last occurrence",
            "Remove all duplicate occurrences",
        ],
    )

    if st.button("Remove Duplicates"):

        keep_mapping = {
            "Keep first occurrence": "first",
            "Keep last occurrence": "last",
            "Remove all duplicate occurrences": False,
        }

        st.session_state.dataset = remove_duplicates(
            df,
            keep=keep_mapping[keep_option],
        )

        st.success("Duplicate removal completed.")
        st.rerun()

    st.divider()

    st.subheader("Handle Missing Values")

    missing_strategy = st.selectbox(
        "Choose strategy",
        [
            "drop_rows",
            "drop_columns",
            "fill_value",
            "mean",
            "median",
            "mode",
        ],
    )

    selected_missing_columns = st.multiselect(
        "Select columns",
        df.columns.tolist(),
        default=df.columns.tolist(),
    )

    fill_value = None

    if missing_strategy == "fill_value":

        fill_value = st.text_input(
            "Value to replace missing values"
        )

    if st.button("Apply Missing Value Strategy"):

        try:

            st.session_state.dataset = handle_missing_values(
                df,
                strategy=missing_strategy,
                fill_value=fill_value,
                columns=selected_missing_columns,
            )

            st.success("Missing-value handling completed.")
            st.rerun()

        except Exception as error:

            st.error(str(error))

    st.divider()

    st.subheader("Drop Columns")

    columns_to_drop = st.multiselect(
        "Select columns to remove",
        df.columns.tolist(),
        key="drop_columns_select",
    )

    if st.button("Drop Selected Columns"):

        if not columns_to_drop:

            st.warning("Select at least one column.")

        else:

            st.session_state.dataset = drop_columns(
                df,
                columns_to_drop,
            )

            st.success("Columns removed.")
            st.rerun()

    st.divider()

    st.subheader("Rename a Column")

    rename_column = st.selectbox(
        "Column to rename",
        df.columns.tolist(),
        key="rename_column_select",
    )

    new_column_name = st.text_input(
        "New column name"
    )

    if st.button("Rename Column"):

        if not new_column_name.strip():

            st.warning("Enter a new column name.")

        else:

            try:

                st.session_state.dataset = rename_columns(
                    df,
                    {rename_column: new_column_name.strip()},
                )

                st.success("Column renamed.")
                st.rerun()

            except Exception as error:

                st.error(str(error))

    st.divider()

    st.subheader("Convert Data Type")

    type_column = st.selectbox(
        "Select column",
        df.columns.tolist(),
        key="type_column_select",
    )

    new_dtype = st.selectbox(
        "Convert to",
        [
            "string",
            "int",
            "float",
            "datetime",
        ],
    )

    if st.button("Convert Data Type"):

        try:

            st.session_state.dataset = convert_column_types(
                df,
                {type_column: new_dtype},
            )

            st.success("Data type converted.")
            st.rerun()

        except Exception as error:

            st.error(str(error))

    st.divider()

    st.subheader("Current Dataset")

    st.dataframe(
        st.session_state.dataset,
        width="stretch",
    )


# =========================================================
# VISUALIZATION PAGE
# =========================================================

elif page == "Visualization":

    st.markdown('<div class="section-label">Visual Intelligence</div>',
                unsafe_allow_html=True)

    st.header("Visualization Studio")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        [
            "Histogram",
            "Bar Chart",
            "Scatter Plot",
            "Line Chart",
            "Box Plot",
        ],
    )

    if chart_type == "Histogram":

        if not numeric_columns:

            st.warning("No numerical columns available.")

        else:

            x_column = st.selectbox(
                "Select numerical column",
                numeric_columns,
            )

            fig = px.histogram(
                df,
                x=x_column,
                title=f"Distribution of {x_column}",
                template="plotly_dark",
                color_discrete_sequence=["#a78bfa"],
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    elif chart_type == "Bar Chart":

        available_columns = (
            categorical_columns + numeric_columns
        )

        if not available_columns:

            st.warning("No suitable columns available.")

        else:

            x_column = st.selectbox(
                "Category / X-axis",
                available_columns,
            )

            if numeric_columns:

                y_column = st.selectbox(
                    "Numerical Y-axis",
                    numeric_columns,
                )

                fig = px.bar(
                    df,
                    x=x_column,
                    y=y_column,
                    template="plotly_dark",
                    color_discrete_sequence=["#a78bfa"],
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            else:

                counts = df[x_column].value_counts().reset_index()

                counts.columns = [x_column, "count"]

                fig = px.bar(
                    counts,
                    x=x_column,
                    y="count",
                    template="plotly_dark",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

    elif chart_type in ["Scatter Plot", "Line Chart"]:

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numerical columns are required."
            )

        else:

            x_column = st.selectbox(
                "X-axis",
                numeric_columns,
            )

            y_column = st.selectbox(
                "Y-axis",
                [
                    col for col in numeric_columns
                    if col != x_column
                ],
            )

            if chart_type == "Scatter Plot":

                fig = px.scatter(
                    df,
                    x=x_column,
                    y=y_column,
                    template="plotly_dark",
                    color_discrete_sequence=["#a78bfa"],
                )

            else:

                fig = px.line(
                    df,
                    x=x_column,
                    y=y_column,
                    template="plotly_dark",
                )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    elif chart_type == "Box Plot":

        if not numeric_columns:

            st.warning("No numerical columns available.")

        else:

            selected_column = st.selectbox(
                "Select numerical column",
                numeric_columns,
            )

            fig = px.box(
                df,
                y=selected_column,
                template="plotly_dark",
                color_discrete_sequence=["#a78bfa"],
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )


# =========================================================
# EXPORT PAGE
# =========================================================

elif page == "Export":

    st.markdown('<div class="section-label">Data Delivery</div>',
                unsafe_allow_html=True)

    st.header("Export Workspace")

    st.write(
        "Download your current working dataset "
        "after cleaning and transformations."
    )

    st.subheader("Dataset Summary")

    st.metric(
        "Records",
        f"{len(df):,}"
    )

    st.metric(
        "Features",
        len(df.columns)
    )

    st.dataframe(
        df.head(10),
        width="stretch",
    )

    st.divider()

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name="data_agent_cleaned.csv",
        mime="text/csv",
    )

    import io

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data",
        )

    st.download_button(
        label="Download as Excel",
        data=excel_buffer.getvalue(),
        file_name="data_agent_cleaned.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#657086;font-size:11px;">
    ◈ DATA AGENT &nbsp; | &nbsp;
    AI-POWERED DATA INTELLIGENCE
</div>
""", unsafe_allow_html=True)
