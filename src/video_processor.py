import os
import time
import cv2


def process_video(
    video_path,
    model,
    evidence_dir,
    confidence_threshold=0.40,
    frame_skip=10,
    min_event_gap_seconds=3
):
    """
    Process an aerial video using YOLO.

    Returns actual video-processing measurements:
    - total frames
    - frames analyzed
    - detections
    - events
    - processing time
    - analyzed FPS
    - real-time factor
    """

    os.makedirs(
        evidence_dir,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise ValueError(
            "Could not open the uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration_seconds = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    events = []

    frame_number = 0
    frames_processed = 0
    detections_found = 0

    last_event_time = -9999.0

    start_time = time.perf_counter()

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Process every Nth frame
        if frame_number % frame_skip != 0:
            frame_number += 1
            continue

        frames_processed += 1

        timestamp_seconds = (
            frame_number / fps
        )

        results = model(
            frame,
            conf=confidence_threshold,
            verbose=False
        )

        result = results[0]

        frame_detections = []

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

                frame_detections.append(
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

        detections_found += len(
            frame_detections
        )

        # -------------------------------------------------
        # Event grouping
        # -------------------------------------------------

        if (
            frame_detections
            and
            (
                timestamp_seconds
                - last_event_time
                >= min_event_gap_seconds
            )
        ):

            annotated = result.plot()

            evidence_filename = (
                f"video_event_"
                f"{len(events) + 1:03d}.jpg"
            )

            evidence_path = os.path.join(
                evidence_dir,
                evidence_filename
            )

            cv2.imwrite(
                evidence_path,
                annotated
            )

            events.append(
                {
                    "detections":
                        frame_detections,

                    "evidence_path":
                        evidence_path,

                    "frame_number":
                        frame_number,

                    "timestamp_seconds":
                        round(
                            timestamp_seconds,
                            2
                        ),

                    "width":
                        width,

                    "height":
                        height
                }
            )

            last_event_time = (
                timestamp_seconds
            )

        frame_number += 1

    cap.release()

    processing_time = (
        time.perf_counter()
        - start_time
    )

    # -----------------------------------------------------
    # Correct performance metrics
    # -----------------------------------------------------

    analyzed_fps = (
        frames_processed / processing_time
        if processing_time > 0
        else 0
    )

    realtime_factor = (
        duration_seconds / processing_time
        if processing_time > 0
        else 0
    )

    return {
        "events": events,

        "frames_processed":
            frames_processed,

        "detections_found":
            detections_found,

        "total_frames":
            total_frames,

        "duration_seconds":
            round(
                duration_seconds,
                2
            ),

        "processing_time_seconds":
            round(
                processing_time,
                2
            ),

        "analyzed_fps":
            round(
                analyzed_fps,
                2
            ),

        "realtime_factor":
            round(
                realtime_factor,
                2
            ),

        "frame_skip":
            frame_skip
    }