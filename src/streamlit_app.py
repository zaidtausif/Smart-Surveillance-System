import os
import base64
import pandas as pd
import streamlit as st
from PIL import Image

# --- Configuration ---
LOG_PATH = os.path.join("outputs", "alerts", "log.csv")
BASE_PATH = r"D:\Honeywell"  # Base directory prefix for finding images

# --- Page Setup ---
st.set_page_config(page_title="AI Surveillance Dashboard", layout="wide")
st.title("AI Surveillance Alerts")

# --- Check for Log File ---
if not os.path.exists(LOG_PATH):
    st.warning("No log file found. Please run the detection script to generate alerts.")
    st.stop()

# --- Load and Prepare Data ---
try:
    df = pd.read_csv(LOG_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["video_id"] = df["source_folder"].astype(str) + " | " + df["source_video"].astype(str)
    video_options = sorted(df["video_id"].unique().tolist())
except Exception as e:
    st.error(f"Error loading or processing the log file: {e}")
    st.stop()

# --- Initialize Session State ---
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "preview_img_path" not in st.session_state:
    st.session_state.preview_img_path = None

# --- Top-Screen Snapshot Overlay ---
if st.session_state.preview_img_path:
    if os.path.exists(st.session_state.preview_img_path):
        with open(st.session_state.preview_img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()

        # Top-screen overlay with full width coverage
        st.markdown(f"""
        <style>
        .top-overlay {{
            position: fixed;
            top: 0; 
            left: 0;
            width: 100%; 
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 9999;
            overflow-y: auto;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding-top: 20px;
        }}
        .top-overlay-content {{
            width: 100%;
            max-width: 1200px;
            background: linear-gradient(135deg, rgba(44, 62, 80, 0.95) 0%, rgba(52, 73, 94, 0.95) 100%);
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
            position: relative;
            border-radius: 0 0 15px 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
            backdrop-filter: blur(10px);
        }}
        .overlay-header {{
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding: 0 10px;
        }}
        .overlay-title {{
            color: #ffffff;
            font-size: 28px;
            font-weight: bold;
            margin: 0;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.7);
        }}
        .overlay-close-btn {{
            background: #e74c3c;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }}
        .overlay-close-btn:hover {{
            background: #c0392b;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 76, 60, 0.6);
        }}
        .overlay-image-container {{
            width: 85%;
            max-width: 1000px;
            height: 65vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: transparent;
            border-radius: 15px;
            padding: 20px;
            position: relative;
        }}
        .overlay-image-container img {{
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            border-radius: 12px;
            object-fit: contain;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
            border: 2px solid rgba(255, 255, 255, 0.2);
        }}
        .overlay-info {{
            color: #ffffff;
            margin-top: 20px;
            text-align: center;
            font-size: 16px;
            font-weight: 500;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.7);
        }}
        </style>

        <div class="top-overlay" onclick="document.getElementById('close-overlay-btn').click()">
            <div class="top-overlay-content" onclick="event.stopPropagation()">
                <div class="overlay-header">
                    <h2 class="overlay-title">📸 Snapshot Preview</h2>
                    <button class="overlay-close-btn" onclick="document.getElementById('close-overlay-btn').click()">
                        ✖ Close
                    </button>
                </div>
                <div class="overlay-image-container">
                    <img src="data:image/png;base64,{img_base64}" alt="Alert Snapshot" />
                </div>
                <div class="overlay-info">
                    Click anywhere outside the image or use the Close button to return to the dashboard
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hidden Streamlit close button for JavaScript interaction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔙 Close Preview", key="close_overlay", use_container_width=True, type="primary"):
                st.session_state.preview_img_path = None
                st.rerun()
            
        # JavaScript to handle overlay close
        st.components.v1.html("""
        <script>
        document.getElementById = function(id) {
            if (id === 'close-overlay-btn') {
                return {
                    click: function() {
                        // Find and click the Streamlit close button
                        const buttons = window.parent.document.querySelectorAll('button');
                        for (let button of buttons) {
                            if (button.textContent.includes('Close Preview')) {
                                button.click();
                                break;
                            }
                        }
                    }
                };
            }
            return null;
        };
        </script>
        """, height=0)
    else:
        st.session_state.preview_img_path = None

# --- Main App Logic ---
if st.session_state.selected_video is None:
    st.subheader("Available Videos with Alerts")
    for vid in video_options:
        if st.button(f"▶ {vid}", key=vid, use_container_width=True):
            st.session_state.selected_video = vid
            st.rerun()
else:
    # --- Header and Video Switcher ---
    st.subheader("Alert Details")
    chosen_video = st.selectbox(
        "Switch Video",
        video_options,
        index=video_options.index(st.session_state.selected_video)
    )
    if chosen_video != st.session_state.selected_video:
        st.session_state.selected_video = chosen_video
        st.session_state.preview_img_path = None
        st.rerun()

    # Filter dataframe for the selected video
    folder, video = st.session_state.selected_video.split(" | ", 1)
    fdf = df[(df["source_folder"] == folder) & (df["source_video"] == video)]

    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("Filters")
        types = ["All"] + sorted(fdf["type"].dropna().unique().tolist())
        labels = ["All"] + sorted(fdf["object_label"].dropna().unique().tolist())
        tsel = st.selectbox("Alert Type", types, index=0)
        lsel = st.selectbox("Object Label", labels, index=0)

    if tsel != "All":
        fdf = fdf[fdf["type"] == tsel]
    if lsel != "All":
        fdf = fdf[fdf["object_label"] == lsel]

    # --- Merge Consecutive Duplicates ---
    fdf = fdf.sort_values("timestamp").reset_index(drop=True)
    group_keys = ["source_folder", "source_video", "object_label", "track_id"]
    if not fdf.empty:
        fdf['group_key'] = fdf[group_keys].ne(fdf[group_keys].shift()).any(axis=1).cumsum()
        fdf = fdf.groupby('group_key').first().reset_index(drop=True)

    st.write(f"Showing **{len(fdf)}** unique alerts for *{st.session_state.selected_video}*")
    st.markdown("---")

    # --- Display Alerts in a Custom Table Layout ---
    header_cols = st.columns([3, 2, 2, 1, 1, 2, 2])
    header_fields = ["Timestamp", "Alert Type", "Object", "Track ID", "Frame", "Score", "Snapshot"]
    for col, field in zip(header_cols, header_fields):
        col.markdown(f"**{field}**")

    for index, row in fdf.iterrows():
        row_cols = st.columns([3, 2, 2, 1, 1, 2, 2])
        row_cols[0].write(row["timestamp"].strftime('%Y-%m-%d %H:%M:%S'))
        row_cols[1].write(row["type"])
        row_cols[2].write(row["object_label"])
        row_cols[3].write(str(row["track_id"]))
        row_cols[4].write(str(row["track_id"]))
        row_cols[4].write(str(row["frame"]))
        row_cols[5].write(f"{row['score']:.2f}" if pd.notna(row['score']) else "N/A")

        with row_cols[6]:
            snap_path = row["snap_path"]
            if pd.notna(snap_path) and isinstance(snap_path, str):
                full_path = os.path.join(BASE_PATH, snap_path) if not os.path.isabs(snap_path) else snap_path
                if os.path.exists(full_path):
                    if st.button("View Screenshot", key=f"view_{index}", use_container_width=True):
                        st.session_state.preview_img_path = full_path
                        st.rerun()
                    st.image(full_path, width=150)
                else:
                    st.caption("Image not found")
            else:
                st.caption("No snapshot")

    st.markdown("---")
    if st.button("⬅️ Back to Video List"):
        st.session_state.selected_video = None
        st.session_state.preview_img_path = None
        st.rerun()