import os
import base64
import tempfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from src.incidents import create_incident, get_all_incidents, update_incident_status
from src.severity import calculate_severity, recommended_action
from src.video_processor import process_video


MODEL_PATH = os.path.join("models", "aerominds_dumping_v2.pt")
CONFIDENCE_THRESHOLD = 0.40


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def collect_detections(result, model):
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        detections.append(
            {
                "class": model.names.get(class_id, str(class_id)),
                "confidence": round(confidence * 100, 2),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )
    return detections


def show_severity(severity, action):
    st.markdown('<div class="section-card"><h2>Incident Severity</h2>', unsafe_allow_html=True)
    columns = st.columns(3)
    columns[0].metric("Level", severity["level"])
    columns[1].metric("Coverage", f'{severity["coverage_percent"]}%')
    columns[2].metric("Clusters", severity["cluster_count"])
    st.info(f'{severity["reason"]} Recommended action: {action}.')
    st.markdown('</div>', unsafe_allow_html=True)


def save_upload(uploaded_file, suffix):
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded_file.getvalue())
    handle.close()
    return handle.name


def image_detection(model, uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    result = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = collect_detections(result, model)
    severity = calculate_severity(
        detections=detections,
        image_width=result.orig_shape[1],
        image_height=result.orig_shape[0],
    )
    action = recommended_action(severity["level"])

    st.image(result.plot()[:, :, ::-1], caption="Detection result", use_container_width=True)
    if detections:
        st.dataframe(detections, use_container_width=True, hide_index=True)
        create_incident(
            event_class=max(detections, key=lambda item: item["confidence"])["class"],
            confidence=max(detections, key=lambda item: item["confidence"])["confidence"],
            severity=severity["level"],
            source_file=uploaded_file.name,
            evidence_url="Streamlit session result",
            recommended_action=action,
        )
    else:
        st.warning("No dumping site detected above the 40% confidence threshold.")
    show_severity(severity, action)


def video_detection(model, uploaded_file):
    video_path = save_upload(uploaded_file, os.path.splitext(uploaded_file.name)[1])
    evidence_dir = tempfile.mkdtemp(prefix="aerominds_evidence_")
    try:
        with st.spinner("Processing video..."):
            video_result = process_video(
                video_path=video_path,
                model=model,
                evidence_dir=evidence_dir,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                frame_skip=10,
                min_event_gap_seconds=3,
            )
    finally:
        os.unlink(video_path)

    metrics = {
        "Total frames": video_result["total_frames"],
        "Frames processed": video_result["frames_processed"],
        "Detections": video_result["detections_found"],
        "Events": len(video_result["events"]),
        "Processing seconds": video_result["processing_time_seconds"],
    }
    st.json(metrics)
    for index, event in enumerate(video_result["events"], start=1):
        st.subheader(f"Evidence event {index} at {event['timestamp_seconds']} seconds")
        st.image(event["evidence_path"], use_container_width=True)
        detections = event["detections"]
        severity = calculate_severity(detections, event["width"], event["height"])
        action = recommended_action(severity["level"])
        create_incident(
            event_class=max(detections, key=lambda item: item["confidence"])["class"],
            confidence=max(detections, key=lambda item: item["confidence"])["confidence"],
            severity=severity["level"],
            source_file=uploaded_file.name,
            evidence_url=event["evidence_path"],
            recommended_action=action,
        )
        st.dataframe(detections, use_container_width=True, hide_index=True)
        show_severity(severity, action)


def show_incidents():
    st.markdown('<div class="section-card"><h2>Incident History</h2>', unsafe_allow_html=True)
    incidents = get_all_incidents()
    if not incidents:
        st.caption("No incidents recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    for incident in incidents:
        with st.expander(f'{incident["incident_id"]} | {incident["severity"]} | {incident["event_class"]}'):
            st.write(incident)
            new_status = st.selectbox(
                "Status",
                ["PENDING", "ASSIGNED", "CLEARED"],
                index=["PENDING", "ASSIGNED", "CLEARED"].index(incident["status"]),
                key=f'status_{incident["incident_id"]}',
            )
            if st.button("Update status", key=f'update_{incident["incident_id"]}'):
                update_incident_status(incident["incident_id"], new_status)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="AeroMinds", page_icon="🌿", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #0f172a, #111827); color: #ffffff; }
        [data-testid="stHeader"] { background: #0f172a; }
        .block-container { max-width: 1250px; padding-top: 2rem; }
        .brand { display: flex; align-items: center; gap: 18px; padding: 0 0 24px; border-bottom: 1px solid #334155; }
        .brand h1 { margin: 0; color: #ffffff; font-size: 38px; }
        .brand p { margin: 6px 0 0; color: #94a3b8; }
        .logo { width: 74px; height: 74px; object-fit: contain; }
        .section-card { background: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 1rem 1.25rem; margin: 1rem 0; }
        .section-card h2 { color: #ffffff; margin: 0 0 1rem; }
        .status-card { background: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 1.2rem; margin: 1rem 0; }
        .status-pill { display: inline-block; padding: 8px 15px; border-radius: 999px; background: #064e3b; color: #6ee7b7; font-weight: 700; font-size: 13px; }
        [data-testid="stMetric"] { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 12px; }
        [data-testid="stMetricLabel"] { color: #94a3b8; }
        [data-testid="stMetricValue"] { color: #ffffff; }
        [data-testid="stFileUploader"] { background: #0f172a; border: 2px dashed #475569; border-radius: 12px; padding: 1rem; }
        .stButton > button { background: #2563eb; color: #ffffff; border: 0; border-radius: 8px; font-weight: 700; }
        .stButton > button:hover { background: #1d4ed8; color: #ffffff; }
        [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #334155; }
        div[data-baseweb="tab-list"] { gap: 8px; }
        button[data-baseweb="tab"] { color: #cbd5e1; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #60a5fa; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = os.path.join("static", "aerominds_logo.png")
    logo_markup = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as logo_file:
            logo_data = base64.b64encode(logo_file.read()).decode("ascii")
        logo_markup = f'<img class="logo" src="data:image/png;base64,{logo_data}" alt="AeroMinds logo">'
    st.markdown(
        f'<div class="brand">{logo_markup}<div><h1>AeroMinds</h1><p>AI-Powered Drone-Based Waste Detection and Smart Sanitation Response</p></div></div>',
        unsafe_allow_html=True,
    )

    model = load_model()
    incidents = get_all_incidents()
    counts = {
        "Total Incidents": len(incidents),
        "High Severity": sum(item["severity"] == "HIGH" for item in incidents),
        "Medium Severity": sum(item["severity"] == "MEDIUM" for item in incidents),
        "Low Severity": sum(item["severity"] == "LOW" for item in incidents),
        "Pending": sum(item["status"] == "PENDING" for item in incidents),
        "Assigned": sum(item["status"] == "ASSIGNED" for item in incidents),
        "Cleared": sum(item["status"] == "CLEARED" for item in incidents),
    }
    st.markdown('<div class="status-card"><span class="status-pill">AI MODEL ONLINE</span><p><b>Loaded Model:</b> aerominds_dumping_v2.pt</p><p><b>Detected Class:</b> dumping-sites</p></div>', unsafe_allow_html=True)
    metric_columns = st.columns(7)
    for column, (label, value) in zip(metric_columns, counts.items()):
        column.metric(label, value)

    image_tab, video_tab, history_tab = st.tabs(["Image analysis", "Video analysis", "Incident history"])

    with image_tab:
        st.markdown('<div class="section-card"><h2>Upload Aerial Image</h2><p>Upload an aerial or drone image to detect potential dumping sites.</p>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_image and st.button("Analyze image", type="primary"):
            image_detection(model, uploaded_image)
        st.markdown('</div>', unsafe_allow_html=True)

    with video_tab:
        st.markdown('<div class="section-card"><h2>Upload Aerial Video</h2><p>Upload an aerial/drone video for frame-by-frame dumping-site analysis.</p>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Upload an aerial video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video and st.button("Analyze video", type="primary"):
            video_detection(model, uploaded_video)
        st.markdown('</div>', unsafe_allow_html=True)

    with history_tab:
        show_incidents()


if __name__ == "__main__":
    main()
