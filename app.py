from flask import (
    Flask,
    request,
    render_template,
    url_for,
    send_from_directory,
    redirect
)

from ultralytics import YOLO
from werkzeug.utils import secure_filename

import os

from src.severity import (
    calculate_severity,
    recommended_action
)

from src.incidents import (
    create_incident,
    update_incident_status,
    get_all_incidents
)

from src.video_processor import process_video


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

VIDEO_EVIDENCE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "video_evidence"
)

MODEL_PATH = os.path.join(
    "models",
    "aerominds_dumping_v2.pt"
)

ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv"
}

# Maximum upload size: 100 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    VIDEO_EVIDENCE_FOLDER,
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"AeroMinds model not found: {MODEL_PATH}"
    )


print("=" * 60)
print("Loading AeroMinds model...")
print(f"Model path: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("AeroMinds model loaded successfully!")
print("=" * 60)


# =========================================================
# FILE VALIDATION HELPERS
# =========================================================

def allowed_image_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_IMAGE_EXTENSIONS
    )


def allowed_video_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_VIDEO_EXTENSIONS
    )

# =========================================================
# DASHBOARD STATISTICS
# =========================================================

@app.context_processor
def inject_dashboard_stats():

    incidents = get_all_incidents()

    high_count = sum(
        1
        for item in incidents
        if item["severity"] == "HIGH"
    )

    medium_count = sum(
        1
        for item in incidents
        if item["severity"] == "MEDIUM"
    )

    low_count = sum(
        1
        for item in incidents
        if item["severity"] == "LOW"
    )

    pending_count = sum(
        1
        for item in incidents
        if item["status"] == "PENDING"
    )

    assigned_count = sum(
        1
        for item in incidents
        if item["status"] == "ASSIGNED"
    )

    cleared_count = sum(
        1
        for item in incidents
        if item["status"] == "CLEARED"
    )

    return {
        "dashboard_stats": {
            "total": len(incidents),
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "pending": pending_count,
            "assigned": assigned_count,
            "cleared": cleared_count
        }
    }

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",

        model_name=os.path.basename(
            MODEL_PATH
        ),

        incidents=get_all_incidents()
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "application": "AeroMinds",
        "model": os.path.basename(
            MODEL_PATH
        ),
        "model_loaded": True
    }


# =========================================================
# IMAGE DETECTION
# =========================================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    # -----------------------------------------------------
    # Validate upload
    # -----------------------------------------------------

    if "image" not in request.files:

        return (
            "No image uploaded.",
            400
        )


    file = request.files["image"]


    if file.filename == "":

        return (
            "No image selected.",
            400
        )


    if not allowed_image_file(
        file.filename
    ):

        return (
            "Unsupported image format. "
            "Use JPG, JPEG, PNG or WEBP.",
            400
        )


    # -----------------------------------------------------
    # Secure filename
    # -----------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    # -----------------------------------------------------
    # Save upload
    # -----------------------------------------------------

    file.save(input_path)

    print(
        f"Running inference on: "
        f"{input_path}"
    )


    # -----------------------------------------------------
    # YOLO inference
    # -----------------------------------------------------

    results = model(
        input_path,
        conf=0.40
    )


    result = results[0]


    # -----------------------------------------------------
    # Original image dimensions
    # -----------------------------------------------------

    image_height, image_width = (
        result.orig_shape
    )


    # -----------------------------------------------------
    # Collect detections
    # -----------------------------------------------------

    detections = []


    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names.get(
                cls_id,
                str(cls_id)
            )

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0].tolist()
            )

            detections.append(
                {
                    "class": class_name,

                    "confidence": round(
                        confidence * 100,
                        2
                    ),

                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                }
            )


    # -----------------------------------------------------
    # Severity calculation
    # -----------------------------------------------------

    severity = calculate_severity(
        detections=detections,
        image_width=image_width,
        image_height=image_height
    )


    # -----------------------------------------------------
    # Recommended action
    # -----------------------------------------------------

    action = recommended_action(
        severity["level"]
    )


    # -----------------------------------------------------
    # Save annotated image
    # -----------------------------------------------------

    result.save(
        filename=output_path
    )


    # -----------------------------------------------------
    # Create incident
    # -----------------------------------------------------

    incident = None


    if detections:

        primary_detection = max(
            detections,
            key=lambda d: d["confidence"]
        )


        evidence_url = url_for(
            "static_output",
            filename=filename
        )


        incident = create_incident(

            event_class=primary_detection[
                "class"
            ],

            confidence=primary_detection[
                "confidence"
            ],

            severity=severity[
                "level"
            ],

            source_file=filename,

            evidence_url=evidence_url,

            zone_id="Zone A",

            nearest_landmark=(
                "Demo / Manually Configured"
            ),

            recommended_action=action
        )


    # -----------------------------------------------------
    # Render result
    # -----------------------------------------------------

    return render_template(

        "index.html",

        result_image=url_for(
            "static_output",
            filename=filename
        ),

        detections=detections,

        model_name=os.path.basename(
            MODEL_PATH
        ),

        severity=severity,

        recommended_action=action,

        incident=incident,

        incidents=get_all_incidents()
    )


# =========================================================
# VIDEO DETECTION
# =========================================================

@app.route(
    "/detect/video",
    methods=["POST"]
)
def detect_video():

    # -----------------------------------------------------
    # Validate upload
    # -----------------------------------------------------

    if "video" not in request.files:

        return (
            "No video uploaded.",
            400
        )


    file = request.files["video"]


    if file.filename == "":

        return (
            "No video selected.",
            400
        )


    if not allowed_video_file(
        file.filename
    ):

        return (
            "Unsupported video format. "
            "Use MP4, AVI, MOV or MKV.",
            400
        )


    # -----------------------------------------------------
    # Secure filename
    # -----------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    video_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # -----------------------------------------------------
    # Save video
    # -----------------------------------------------------

    file.save(video_path)

    print(
        f"Processing video: "
        f"{video_path}"
    )


    # -----------------------------------------------------
    # Process video
    # -----------------------------------------------------

    try:

        video_result = process_video(

            video_path=video_path,

            model=model,

            evidence_dir=VIDEO_EVIDENCE_FOLDER,

            confidence_threshold=0.40,

            frame_skip=10,

            min_event_gap_seconds=3
        )


    except Exception as exc:

        print(
            f"Video processing error: {exc}"
        )

        return (
            "Video processing failed: "
            f"{exc}",
            500
        )


    # -----------------------------------------------------
    # Create incidents from video events
    # -----------------------------------------------------

    created_incidents = []


    for event in video_result["events"]:

        event_detections = event[
            "detections"
        ]


        if not event_detections:
            continue


        severity = calculate_severity(

            detections=event_detections,

            image_width=event[
                "width"
            ],

            image_height=event[
                "height"
            ]
        )


        action = recommended_action(
            severity["level"]
        )


        primary_detection = max(

            event_detections,

            key=lambda d:
                d["confidence"]
        )


        evidence_filename = os.path.basename(
            event["evidence_path"]
        )


        evidence_url = url_for(

            "video_evidence",

            filename=evidence_filename
        )


        incident = create_incident(

            event_class=
                primary_detection["class"],

            confidence=
                primary_detection["confidence"],

            severity=
                severity["level"],

            source_file=filename,

            evidence_url=evidence_url,

            zone_id="Zone A",

            nearest_landmark=(
                "Demo / Manually Configured"
            ),

            recommended_action=action
        )


        created_incidents.append(
            incident
        )


        # Add URL directly to the event
        # for cleaner dashboard handling.
        event["evidence_url"] = (
            evidence_url
        )


    # -----------------------------------------------------
    # Render video results
    # -----------------------------------------------------

    return render_template(

        "index.html",

        model_name=os.path.basename(
            MODEL_PATH
        ),

        incidents=get_all_incidents(),

        video_result=video_result,

        video_incidents=created_incidents
    )


# =========================================================
# VIDEO EVIDENCE
# =========================================================

@app.route(
    "/video-evidence/<filename>"
)
def video_evidence(filename):

    return send_from_directory(

        VIDEO_EVIDENCE_FOLDER,

        filename
    )


# =========================================================
# INCIDENT STATUS
# =========================================================

@app.route(
    "/incidents/<incident_id>/status",
    methods=["GET", "POST"]
)
def change_incident_status(
    incident_id
):

    # -----------------------------------------------------
    # Direct GET access
    # -----------------------------------------------------

    if request.method == "GET":

        return redirect(
            url_for(
                "view_incident",
                incident_id=incident_id
            )
        )


    # -----------------------------------------------------
    # POST from status buttons
    # -----------------------------------------------------

    new_status = request.form.get(
        "status",
        ""
    )


    incident = update_incident_status(

        incident_id,

        new_status
    )


    if incident is None:

        return (
            "Incident not found "
            "or invalid status.",
            400
        )


    return redirect(

        url_for(

            "view_incident",

            incident_id=incident_id
        )
    )


# =========================================================
# INCIDENT VIEW
# =========================================================

@app.route(
    "/incidents/<incident_id>"
)
def view_incident(
    incident_id
):

    incidents = get_all_incidents()

    incident = None


    for item in incidents:

        if item[
            "incident_id"
        ] == incident_id:

            incident = item

            break


    if incident is None:

        return (
            "Incident not found.",
            404
        )


    return render_template(

        "index.html",

        model_name=os.path.basename(
            MODEL_PATH
        ),

        result_image=incident[
            "evidence_url"
        ],

        detections=[],

        severity={
            "level": incident[
                "severity"
            ],

            "coverage_percent": 0,

            "cluster_count": 0,

            "reason": (
                "Incident generated "
                "from stored detection."
            )
        },

        recommended_action=incident[
            "recommended_action"
        ],

        incident=incident,

        incidents=incidents
    )


# =========================================================
# INCIDENT HISTORY
# =========================================================

@app.route(
    "/incidents"
)
def incidents():

    return render_template(

        "index.html",

        model_name=os.path.basename(
            MODEL_PATH
        ),

        incidents=get_all_incidents()
    )


# =========================================================
# SERVE IMAGE OUTPUTS
# =========================================================

@app.route(
    "/outputs/<filename>"
)
def static_output(filename):

    return send_from_directory(

        OUTPUT_FOLDER,

        filename
    )


# =========================================================
# ERROR HANDLER — LARGE FILE
# =========================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return (
        "Uploaded file is too large. "
        "Maximum allowed size is 100 MB.",
        413
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True
    )