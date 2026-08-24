import os
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
    st.subheader("Incident severity")
    columns = st.columns(3)
    columns[0].metric("Level", severity["level"])
    columns[1].metric("Coverage", f'{severity["coverage_percent"]}%')
    columns[2].metric("Clusters", severity["cluster_count"])
    st.info(f'{severity["reason"]} Recommended action: {action}.')


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
    st.subheader("Incident history")
    incidents = get_all_incidents()
    if not incidents:
        st.caption("No incidents recorded yet.")
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


def main():
    st.set_page_config(page_title="AeroMinds", page_icon="🌿", layout="wide")
    st.title("AeroMinds")
    st.caption("AI-powered aerial dumping-site detection")
    st.sidebar.success("AI model online")
    st.sidebar.write("Model: aerominds_dumping_v2.pt")
    st.sidebar.write("Class: dumping-sites")

    model = load_model()
    image_tab, video_tab, history_tab = st.tabs(["Image analysis", "Video analysis", "Incident history"])

    with image_tab:
        uploaded_image = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_image and st.button("Analyze image", type="primary"):
            image_detection(model, uploaded_image)

    with video_tab:
        uploaded_video = st.file_uploader("Upload an aerial video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video and st.button("Analyze video", type="primary"):
            video_detection(model, uploaded_video)

    with history_tab:
        show_incidents()


if __name__ == "__main__":
    main()
