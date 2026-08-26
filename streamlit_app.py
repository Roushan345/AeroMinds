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
    high_color = "#FF5C64"
    med_color = "#FFB62E"
    low_color = "#35E38A"
    
    color = low_color
    if severity["level"] == "HIGH": color = high_color
    elif severity["level"] == "MEDIUM": color = med_color
    
    html = f"""
    <div class="section-card" style="border-left: 5px solid {color}">
        <h2 style="color: {color}; margin-top: 0; font-size: 24px;">⚠ INCIDENT SEVERITY</h2>
        <div style="font-size: 42px; font-weight: 800; margin-bottom: 20px; color: {color}">{severity["level"]}</div>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <div style="font-size: 28px; font-weight: 700;">{severity["coverage_percent"]}%</div>
                <div style="color: #8EA4B8; font-size: 14px; text-transform: uppercase;">WASTE COVERAGE</div>
            </div>
            <div>
                <div style="font-size: 28px; font-weight: 700;">{severity["cluster_count"]}</div>
                <div style="color: #8EA4B8; font-size: 14px; text-transform: uppercase;">CLUSTER</div>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; margin-top: 20px;">
            <div style="color: #8EA4B8; font-size: 14px; text-transform: uppercase; margin-bottom: 5px;">Recommended Response</div>
            <div style="font-size: 18px; font-weight: 600;">{action}</div>
        </div>
    </div>
    """
    st.html(html)


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

    st.html('<div class="analysis-box" style="margin-top:30px;"><div style="color:#00D9A6; font-weight:700; margin-bottom:15px; display:flex; justify-content:space-between;"><span>AI ANALYSIS COMPLETE</span><span>⚡ SUCCESS</span></div>')
    st.image(result.plot()[:, :, ::-1], use_container_width=True)
    st.html('</div>')

    if detections:
        # Custom HTML table for detections
        html = f"""
        <div style="margin-top: 30px; margin-bottom: 10px; font-weight: 700; color: #F4F7FA; font-size: 20px;">{len(detections)} DETECTIONS</div>
        <table class="styled-table">
        """
        for d in detections:
            html += f'<tr><td style="font-weight: 600;">{d["class"].upper()}</td><td style="color:#4DA3FF; font-weight:bold;">{d["confidence"]}%</td><td style="color:#35E38A;">CONFIRMED</td></tr>'
        html += "</table>"
        st.html(html)
        
        create_incident(
            event_class=max(detections, key=lambda item: item["confidence"])["class"],
            confidence=max(detections, key=lambda item: item["confidence"])["confidence"],
            severity=severity["level"],
            source_file=uploaded_file.name,
            evidence_url="Streamlit session result",
            recommended_action=action,
        )
        show_severity(severity, action)
    else:
        st.warning("No dumping site detected above the 40% confidence threshold.")


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

    # Video Dashboard
    html = f"""
    <div class="section-card">
        <h2 style="font-size: 20px; color: #F4F7FA; margin-bottom: 25px;">VIDEO ANALYSIS</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
            <div>
                <div style="font-size: 32px; font-weight: 700; color: #4DA3FF;">{video_result['processing_time_seconds']}s</div>
                <div style="font-size: 13px; color: #8EA4B8; text-transform: uppercase;">Processing Time</div>
            </div>
            <div>
                <div style="font-size: 32px; font-weight: 700;">{video_result['total_frames']}</div>
                <div style="font-size: 13px; color: #8EA4B8; text-transform: uppercase;">Total Frames</div>
            </div>
            <div>
                <div style="font-size: 32px; font-weight: 700;">{video_result['frames_processed']}</div>
                <div style="font-size: 13px; color: #8EA4B8; text-transform: uppercase;">Frames Processed</div>
            </div>
            <div>
                <div style="font-size: 32px; font-weight: 700; color: #00D9A6;">{len(video_result['events'])}</div>
                <div style="font-size: 13px; color: #8EA4B8; text-transform: uppercase;">Events</div>
            </div>
        </div>
    </div>
    """
    st.html(html)
    
    st.html('<div style="margin-top: 40px; margin-bottom: 20px; font-weight: 700; text-align: center; color: #8EA4B8; letter-spacing: 2px;">──────────── EVIDENCE TIMELINE ────────────</div>')

    for index, event in enumerate(video_result["events"], start=1):
        st.html(f'<div style="font-size: 18px; font-weight: 700; margin-top: 30px; margin-bottom: 10px; color: #F4F7FA;">● EVENT {index} <span style="color: #4DA3FF; margin-left: 10px;">{event["timestamp_seconds"]}s</span></div>')
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
        show_severity(severity, action)


def show_incidents():
    incidents = get_all_incidents()
    if not incidents:
        st.html('<div class="section-card"><p>No incidents recorded yet.</p></div>')
        return
        
    for incident in incidents:
        sev_color = "#35E38A"
        if incident["severity"] == "HIGH": sev_color = "#FF5C64"
        elif incident["severity"] == "MEDIUM": sev_color = "#FFB62E"
        
        status_color = "#35E38A"
        if incident["status"] == "PENDING": status_color = "#FFB62E"
        elif incident["status"] == "ASSIGNED": status_color = "#4DA3FF"

        with st.container():
            st.html(f"""
            <div class="incident-card" style="border-left: 5px solid {sev_color}">
                <div style="display:flex; justify-content:space-between; margin-bottom: 15px;">
                    <div style="font-size: 20px; font-weight: 700; color: #F4F7FA;">INCIDENT {incident['incident_id']}</div>
                    <div style="font-size: 14px; color: #8EA4B8;">{incident['timestamp'].split('.')[0]}</div>
                </div>
                
                <div style="display:flex; justify-content:space-between; margin-bottom: 25px;">
                    <div>
                        <div style="color: {sev_color}; font-weight: 700; font-size: 16px;">{incident['severity']} SEVERITY</div>
                        <div style="font-size: 18px; font-weight: 600; margin-top: 5px;">{incident['event_class']}</div>
                        <div style="color: #4DA3FF; font-weight: 600;">{incident['confidence']}% confidence</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 18px; font-weight: 600;">{incident['zone_id']}</div>
                        <div style="color: #8EA4B8; font-size: 14px; margin-top: 5px;">{incident['nearest_landmark']}</div>
                    </div>
                </div>
                
                <div class="timeline">
                    <div class="timeline-item" style="color: #F4F7FA;">
                        <div class="timeline-dot" style="background: #00D9A6;"></div>
                        DETECTED
                    </div>
                    <div class="timeline-line"></div>
                    <div class="timeline-item" style="color: {'#F4F7FA' if incident['status'] in ['PENDING', 'ASSIGNED', 'CLEARED'] else '#8EA4B8'};">
                        <div class="timeline-dot" style="background: {'#FFB62E' if incident['status'] in ['PENDING', 'ASSIGNED', 'CLEARED'] else '#13283A'};"></div>
                        PENDING
                    </div>
                    <div class="timeline-line"></div>
                    <div class="timeline-item" style="color: {'#F4F7FA' if incident['status'] in ['ASSIGNED', 'CLEARED'] else '#8EA4B8'};">
                        <div class="timeline-dot" style="background: {'#4DA3FF' if incident['status'] in ['ASSIGNED', 'CLEARED'] else '#13283A'};"></div>
                        ASSIGNED
                    </div>
                    <div class="timeline-line"></div>
                    <div class="timeline-item" style="color: {'#F4F7FA' if incident['status'] == 'CLEARED' else '#8EA4B8'};">
                        <div class="timeline-dot" style="background: {'#35E38A' if incident['status'] == 'CLEARED' else '#13283A'};"></div>
                        CLEARED
                    </div>
                </div>
            </div>
            """)
            
            # Use columns for action buttons right under the visual card
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                new_status = st.selectbox(
                    "Update Status",
                    ["PENDING", "ASSIGNED", "CLEARED"],
                    index=["PENDING", "ASSIGNED", "CLEARED"].index(incident["status"]),
                    key=f'status_{incident["incident_id"]}',
                    label_visibility="collapsed"
                )
            with c2:
                if st.button("Apply", key=f'update_{incident["incident_id"]}'):
                    update_incident_status(incident["incident_id"], new_status)
                    st.rerun()


def main():
    st.set_page_config(page_title="AeroMinds", page_icon="🌿", layout="wide")
    st.html("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp { 
            background: #07111F;
            color: #F4F7FA; 
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1250px; padding-top: 1rem; }
        
        /* HER SECTION */
        .hero { 
            padding: 20px 0 30px; 
            border-bottom: 1px solid rgba(255,255,255,0.05); 
            margin-bottom: 20px;
        }
        .hero-top {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 40px;
        }
        .hero-logo {
            display: flex; align-items: center; gap: 15px;
        }
        .logo { width: 120px; height: 120px; object-fit: contain; filter: drop-shadow(0 0 15px rgba(0, 217, 166, 0.4)); }
        .hero-logo h1 { margin: 0; color: #F4F7FA; font-size: 42px; font-weight: 700; }
        
        .system-online {
            display: flex; align-items: center; gap: 8px;
            font-size: 12px; font-weight: 700; color: #8EA4B8; letter-spacing: 1px;
        }
        .online-dot {
            width: 8px; height: 8px; border-radius: 50%; background: #35E38A;
            box-shadow: 0 0 10px #35E38A;
            animation: pulse 1.8s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.35; }
        }
        
        .hero-content {
            margin-bottom: 20px;
        }
        .hero-eyebrow {
            color: #00D9A6; font-size: 14px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px;
        }
        .hero-headline {
            font-size: 54px; font-weight: 800; color: #F4F7FA; margin: 0 0 15px; line-height: 1.1;
        }
        .hero-sub {
            font-size: 20px; color: #8EA4B8; max-width: 600px; line-height: 1.5; font-weight: 300;
        }
        
        /* LIVE STRIP */
        .live-strip {
            display: flex; justify-content: space-between;
            background: #0D1B2A; border: 1px solid #13283A; border-radius: 12px; padding: 15px 25px;
            margin-bottom: 40px;
        }
        .strip-item {
            display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 600; color: #F4F7FA;
        }
        .strip-value { color: #8EA4B8; font-weight: 400; margin-left: 5px; }
        .strip-dot { width: 6px; height: 6px; border-radius: 50%; background: #4DA3FF; box-shadow: 0 0 8px #4DA3FF; }
        
        /* SECTION CARDS */
        .section-card { 
            background: #0D1B2A; 
            border: 1px solid #13283A; 
            border-radius: 16px; 
            padding: 2rem; 
            margin: 1.5rem 0; 
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .section-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(0, 217, 166, 0.10);
            border: 1px solid #1c364e;
        }
        
        /* VISUAL COMMAND CENTER */
        .cmd-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 20px;
        }
        .cmd-box { background: #07111F; border: 1px solid #13283A; border-radius: 12px; padding: 25px; }
        .cmd-title { font-size: 13px; color: #8EA4B8; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 20px; text-align: center; }
        
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 15px; }
        .stat-col { text-align: center; }
        .stat-val { font-size: 38px; font-weight: 700; }
        .stat-lab { font-size: 12px; color: #8EA4B8; font-weight: 600; letter-spacing: 1px; }
        
        .bar-container { width: 100%; display: flex; gap: 4px; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 15px; }
        
        /* UPLOAD DRAG/DROP */
        [data-testid="stFileUploader"] { 
            background: #07111F; 
            border: 2px dashed #13283A; 
            border-radius: 16px; padding: 2rem; 
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border: 2px dashed #00D9A6;
            background: rgba(0, 217, 166, 0.05);
        }
        
        /* RESULTS & TIMELINES */
        .styled-table { width: 100%; border-collapse: collapse; background: #0D1B2A; border-radius: 12px; overflow: hidden; border: 1px solid #13283A; }
        .styled-table td { padding: 15px 20px; border-bottom: 1px solid #13283A; font-size: 15px; }
        .styled-table tr:last-child td { border-bottom: none; }
        
        .analysis-box { background: #0D1B2A; padding: 20px; border-radius: 12px; border: 1px solid #13283A; }
        
        /* INCIDENT CARDS */
        .incident-card {
            background: #0D1B2A; border: 1px solid #13283A; border-radius: 16px; padding: 25px; margin-bottom: 10px;
            transition: all 0.3s ease; box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }
        .incident-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 1px solid #1c364e; }
        
        .timeline { display: flex; justify-content: space-between; align-items: center; margin-top: 25px; padding: 0 10px; }
        .timeline-item { display: flex; flex-direction: column; align-items: center; font-size: 12px; font-weight: 700; letter-spacing: 1px; }
        .timeline-dot { width: 14px; height: 14px; border-radius: 50%; margin-bottom: 10px; border: 2px solid #0D1B2A; box-shadow: 0 0 0 1px #13283A; }
        .timeline-line { flex-grow: 1; height: 2px; background: #13283A; margin: 0 15px; transform: translateY(-10px); }
        
        /* BUTTONS */
        .stButton > button { 
            background: #13283A; color: #F4F7FA; border: 1px solid #1c364e; border-radius: 8px; font-weight: 600; 
            padding: 0.5rem 1.5rem; transition: all 0.2s ease; width: 100%;
        }
        .stButton > button:hover { 
            background: #00D9A6; color: #07111F; border: 1px solid #00D9A6; transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 217, 166, 0.3);
        }
        div[data-testid="stButton"] button p { font-family: 'Outfit', sans-serif; font-size: 15px; }
        
        /* TABS */
        div[data-baseweb="tab-list"] { gap: 15px; background: transparent; padding-bottom: 10px; }
        button[data-baseweb="tab"] { 
            color: #8EA4B8; background: transparent; border: none;
            border-radius: 0; padding: 12px 0; font-weight: 600; font-size: 16px;
            transition: all 0.2s ease; border-bottom: 2px solid transparent !important;
        }
        button[data-baseweb="tab"]:hover { color: #F4F7FA; }
        button[data-baseweb="tab"][aria-selected="true"] { 
            color: #00D9A6; border-bottom: 2px solid #00D9A6 !important; background: transparent;
        }
        
        </style>
        """
    )

    logo_path = os.path.join("static", "aerominds_logo.png")
    logo_markup = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as logo_file:
            logo_data = base64.b64encode(logo_file.read()).decode("ascii")
        logo_markup = f'<img class="logo" src="data:image/png;base64,{logo_data}" alt="AeroMinds logo">'

    # HERO SECTION (1) & STRIP (3)
    st.html(
        f"""
        <div class="hero">
            <div class="hero-top">
                <div class="hero-logo">{logo_markup} <h1>AeroMinds</h1></div>
                <div class="system-online"><div class="online-dot"></div> SYSTEM ONLINE</div>
            </div>
            <div class="hero-content">
                <div class="hero-eyebrow">AI-Powered Aerial Waste Intelligence</div>
                <h2 class="hero-headline">Detect • Assess • Respond</h2>
                <div class="hero-sub">Autonomous aerial intelligence for smarter urban sanitation and real-time environmental protection.</div>
            </div>
        </div>
        <div class="live-strip">
            <div class="strip-item"><div class="strip-dot"></div> AI MODEL ONLINE <span class="strip-value">YOLOv8 • 3.0M PARAMETERS</span></div>
            <div class="strip-item"><div class="strip-dot"></div> INFERENCE READY <span class="strip-value">8.1 GFLOPs</span></div>
            <div class="strip-item"><div class="strip-dot"></div> DETECTION CLASS <span class="strip-value">DUMPING-SITES</span></div>
        </div>
        """
    )

    model = load_model()
    incidents = get_all_incidents()
    total = len(incidents)
    high = sum(1 for item in incidents if item["severity"] == "HIGH")
    medium = sum(1 for item in incidents if item["severity"] == "MEDIUM")
    low = sum(1 for item in incidents if item["severity"] == "LOW")
    pending = sum(1 for item in incidents if item["status"] == "PENDING")
    assigned = sum(1 for item in incidents if item["status"] == "ASSIGNED")
    cleared = sum(1 for item in incidents if item["status"] == "CLEARED")

    # COMMAND CENTER (2 & 6)
    high_pct = (high/total*100) if total > 0 else 0
    med_pct = (medium/total*100) if total > 0 else 0
    low_pct = (low/total*100) if total > 0 else 0
    
    pend_pct = (pending/total*100) if total > 0 else 0
    ass_pct = (assigned/total*100) if total > 0 else 0
    clr_pct = (cleared/total*100) if total > 0 else 0

    st.html(f"""
    <div class="cmd-grid">
        <div class="cmd-box">
            <div class="cmd-title">───── INCIDENT DISTRIBUTION ─────</div>
            <div class="stat-row">
                <div class="stat-col"><div class="stat-val">{total}</div><div class="stat-lab">TOTAL</div></div>
                <div class="stat-col"><div class="stat-val" style="color:#FF5C64;">{high}</div><div class="stat-lab">HIGH</div></div>
                <div class="stat-col"><div class="stat-val" style="color:#FFB62E;">{medium}</div><div class="stat-lab">MEDIUM</div></div>
                <div class="stat-col"><div class="stat-val" style="color:#35E38A;">{low}</div><div class="stat-lab">LOW</div></div>
            </div>
            <div class="bar-container">
                <div style="width: {high_pct}%; background: #FF5C64;"></div>
                <div style="width: {med_pct}%; background: #FFB62E;"></div>
                <div style="width: {low_pct}%; background: #35E38A;"></div>
            </div>
        </div>
        <div class="cmd-box">
            <div class="cmd-title">───── RESPONSE STATUS ─────</div>
            <div class="stat-row">
                <div class="stat-col"><div class="stat-val" style="color:#FFB62E;">{pending}</div><div class="stat-lab">PENDING</div></div>
                <div class="stat-col"><div class="stat-val" style="color:#4DA3FF;">{assigned}</div><div class="stat-lab">ASSIGNED</div></div>
                <div class="stat-col"><div class="stat-val" style="color:#35E38A;">{cleared}</div><div class="stat-lab">CLEARED</div></div>
            </div>
            <div class="bar-container">
                <div style="width: {pend_pct}%; background: #FFB62E;"></div>
                <div style="width: {ass_pct}%; background: #4DA3FF;"></div>
                <div style="width: {clr_pct}%; background: #35E38A;"></div>
            </div>
        </div>
    </div>
    """)

    image_tab, video_tab, history_tab = st.tabs(["[ Analyze Image ]", "[ Analyze Video ]", "[ Timeline ]"])

    with image_tab:
        # UPLOAD AREA (4)
        uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        st.html("""
        <div style="text-align: center; margin-top: 10px;">
            <div style="font-weight: 700; color: #8EA4B8; letter-spacing: 2px; text-transform: uppercase;">DRAG & DROP AERIAL IMAGE</div>
            <div style="font-size: 24px; margin-top: 10px; color: #F4F7FA;">⇧</div>
            <div style="font-weight: 600; color: #F4F7FA; margin-top: 5px;">Drop image here</div>
            <div style="font-size: 13px; color: #8EA4B8; margin-top: 5px;">JPG • PNG • WEBP</div>
        </div>
        """)
        if uploaded_image and st.button("⚡ ANALYZE IMAGE"):
            image_detection(model, uploaded_image)

    with video_tab:
        uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
        st.html("""
        <div style="text-align: center; margin-top: 10px;">
            <div style="font-weight: 700; color: #8EA4B8; letter-spacing: 2px; text-transform: uppercase;">DRAG & DROP AERIAL VIDEO</div>
            <div style="font-size: 24px; margin-top: 10px; color: #F4F7FA;">⇧</div>
            <div style="font-weight: 600; color: #F4F7FA; margin-top: 5px;">Drop video here</div>
            <div style="font-size: 13px; color: #8EA4B8; margin-top: 5px;">MP4 • AVI • MOV</div>
        </div>
        """)
        if uploaded_video and st.button("⚡ ANALYZE VIDEO"):
            video_detection(model, uploaded_video)

    with history_tab:
        show_incidents()


if __name__ == "__main__":
    main()
