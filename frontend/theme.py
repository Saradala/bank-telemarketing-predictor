"""Shared visual theme, pulled from the real Figma design (file VkyEQ4oNoz4UkvI8X9vfEE,
frame "Home - Dashboard", node 2:10479) via get_design_context - these are the project's
real design tokens, not guessed. Reuse COLORS and inject_css() on every page for a consistent look.
"""
import streamlit as st

COLORS = {
    "text_primary": "#23362f",    # headings, large numbers
    "text_secondary": "#65736d",  # labels, captions, body copy
    "accent": "#176547",          # primary buttons, links, step markers
    "accent_bg": "#eaf3ee",       # icon chips, step-marker backgrounds
    "badge_bg": "#eef1ef",        # status pill background
    "border": "#dde4df",          # card borders, dividers
    "surface": "#ffffff",
    "page_bg": "#f5f7f6",
}


def inject_css():
    """Call once near the top of every page, right after st.set_page_config()."""
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Source Sans 3', sans-serif; }}
        .stApp {{ background-color: {COLORS['page_bg']}; }}
        h1, h2, h3 {{ color: {COLORS['text_primary']} !important; }}
        [data-testid="stMetricValue"] {{ color: {COLORS['text_primary']}; }}
        [data-testid="stMetricLabel"] {{ color: {COLORS['text_secondary']}; }}

        .status-badge {{
            display: inline-block; background-color: {COLORS['badge_bg']}; color: {COLORS['text_secondary']};
            padding: 4px 10px; border-radius: 100px; font-size: 13px; font-weight: 600;
        }}
        .card {{
            background-color: {COLORS['surface']}; border: 1px solid {COLORS['border']};
            border-radius: 10px; padding: 22px; height: 100%;
        }}
        .step-marker {{
            background-color: {COLORS['accent_bg']}; color: {COLORS['accent']}; font-weight: 600;
            width: 28px; height: 28px; border-radius: 100px; display: inline-flex;
            align-items: center; justify-content: center; margin-right: 10px;
        }}
        div.stButton > button[kind="primary"] {{
            background-color: {COLORS['accent']}; border-color: {COLORS['accent']};
        }}
        </style>
    """, unsafe_allow_html=True)


def status_badge(text: str) -> str:
    return f'<span class="status-badge">&#9679; {text}</span>'
