# main.py
import os
import tempfile
from typing import List

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

TIMESTEPS = 5
FEATURES_PER_FRAME = 1662  # 468*3 + 33*4 + 21*3 + 21*3

app = FastAPI()

# Allow CORS from anywhere for now (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_holistic = mp.solutions.holistic


def extract_features_from_frame(results: mp_holistic.Holistic, frame_rgb: np.ndarray) -> np.ndarray:
    """
    Take a processed frame (results from holistic) and output
    a (1662,) features array in the same order as Android code:
    face (468*3), pose (33*4), left hand (21*3), right hand (21*3).
    """
    features = np.zeros((FEATURES_PER_FRAME,), dtype=np.float32)
    idx = 0

    # Face: 468 * 3
    face_landmarks = results.face_landmarks.landmark if results.face_landmarks else []
    count = min(468, len(face_landmarks))
    for i in range(count):
        lm = face_landmarks[i]
        features[idx] = lm.x
        features[idx + 1] = lm.y
        features[idx + 2] = lm.z
        idx += 3
    remaining = 468 - count
    if remaining > 0:
        idx += remaining * 3  # already zeros

    # Pose: 33 * 4
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
    count = min(33, len(pose_landmarks))
    for i in range(count):
        lm = pose_landmarks[i]
        features[idx] = lm.x
        features[idx + 1] = lm.y
        features[idx + 2] = lm.z
        # Holistic gives visibility; default to 1.0 if not present
        vis = lm.visibility if hasattr(lm, "visibility") else 1.0
        features[idx + 3] = vis
        idx += 4
    remaining = 33 - count
    if remaining > 0:
        idx += remaining * 4

    # Hands: left & right, each 21 * 3
    # Mediapipe Holistic typically gives left then right
    # If missing, we fill with zeros.

    def add_hand(hand_landmarks_list):
        nonlocal idx
        if hand_landmarks_list:
            hand_lms = hand_landmarks_list.landmark
            count = min(21, len(hand_lms))
            for i in range(count):
                lm = hand_lms[i]
                features[idx] = lm.x
                features[idx + 1] = lm.y
                features[idx + 2] = lm.z
                idx += 3
            remaining = 21 - count
            if remaining > 0:
                idx += remaining * 3

        else:
            # no landmarks for this hand
            idx += 21 * 3  # zeros

    add_hand(results.left_hand_landmarks)
    add_hand(results.right_hand_landmarks)

    assert idx == FEATURES_PER_FRAME, f"Feature length mismatch: {idx} != {FEATURES_PER_FRAME}"
    return features


def extract_sequence_from_video(video_path: str) -> np.ndarray:
    """
    Load video, sample TIMESTEPS frames, run holistic on them,
    and return flattened (TIMESTEPS * FEATURES_PER_FRAME,) array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")

    steps = max(1, TIMESTEPS - 1)
    frame_indices: List[int] = []
    for step in range(TIMESTEPS):
        if TIMESTEPS == 1:
            idx = total_frames // 2
        else:
            idx = int((total_frames - 1) * step / steps)
        frame_indices.append(idx)

    sequence = np.zeros((TIMESTEPS, FEATURES_PER_FRAME), dtype=np.float32)

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:
        for t, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # leave this timestep as zeros
                continue

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            features = extract_features_from_frame(results, frame_rgb)
            sequence[t] = features

    cap.release()
    return sequence.reshape(-1)  # flatten


@app.post("/extract")
async def extract_endpoint(video: UploadFile = File(...)):
    if video.content_type is None or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    try:
        # Save to a temp file
        suffix = os.path.splitext(video.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        try:
            features = extract_sequence_from_video(tmp_path)
            if features.shape[0] != TIMESTEPS * FEATURES_PER_FRAME:
                raise ValueError(
                    f"Unexpected feature size: {features.shape[0]} != {TIMESTEPS * FEATURES_PER_FRAME}"
                )

            return JSONResponse(
                content={"features": features.astype(float).tolist()}
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
