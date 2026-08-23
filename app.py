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


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

MODEL_PATH = os.path.join(
    "models",
    "aerominds_dumping_v2.pt"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# =========================================================
# DIRECTORIES
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
        f"Model not found: {MODEL_PATH}"
    )

print("=" * 60)
print("Loading AeroMinds model...")
print(f"Model path: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("AeroMinds model loaded successfully!")
print("=" * 60)


# =========================================================
# HELPER
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


# =========================================================
# HOME
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
# HEALTH
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
# DETECTION
# =========================================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    # -----------------------------------------------------
    # File validation
    # -----------------------------------------------------

    if "image" not in request.files:
        return "No image uploaded.", 400

    file = request.files["image"]

    if file.filename == "":
        return "No image selected.", 400

    if not allowed_file(file.filename):
        return (
            "Unsupported file type. "
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
        app.config["UPLOAD_FOLDER"],
        filename
    )

    output_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        filename
    )


    # -----------------------------------------------------
    # Save uploaded file
    # -----------------------------------------------------

    file.save(input_path)

    print(
        f"Running inference on: {input_path}"
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
    # Image dimensions
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
    # Severity
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
            event_class=primary_detection["class"],
            confidence=primary_detection["confidence"],
            severity=severity["level"],
            source_file=filename,
            evidence_url=evidence_url,
            zone_id="Zone A",
            nearest_landmark="Demo / Manually Configured",
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
# INCIDENT STATUS
# =========================================================

@app.route(
    "/incidents/<incident_id>/status",
    methods=["GET", "POST"]
)
def change_incident_status(incident_id):

    # If someone opens the URL directly in the browser,
    # redirect them to the incident page.
    if request.method == "GET":

        return redirect(
            url_for(
                "view_incident",
                incident_id=incident_id
            )
        )

    # POST from Pending / Assigned / Cleared button
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
            "Incident not found or invalid status.",
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
def view_incident(incident_id):

    incidents = get_all_incidents()

    incident = None

    for item in incidents:

        if item["incident_id"] == incident_id:

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
            "level": incident["severity"],
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
# OUTPUT FILES
# =========================================================

@app.route(
    "/outputs/<filename>"
)
def static_output(filename):

    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename
    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )