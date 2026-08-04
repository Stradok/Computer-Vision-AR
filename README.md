# Computer Vision AR experiments

## Module 1: Portal (hand gesture)

Trace a circle in the air with your index finger to open a Doctor-Strange
-style portal; make a fist to close it.

Uses MediaPipe's Tasks `HandLandmarker` (the old `mp.solutions.hands` API
was removed in mediapipe 1.0) for hand tracking, and a procedural
rotating-spiral effect (numpy/OpenCV, no external assets) for the portal
visual.

Setup (one-time model download, already done in this checkout):

```
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Run:

```
source .venv/bin/activate
python src/portal.py
```

Add `--show-landmarks` to overlay the 21 hand landmarks (useful for
debugging pose detection). Press `q` to quit.

Gesture logic lives in `src/portal.py` (`PortalState`): it watches for a
"pointing" pose (only the index finger extended), records the fingertip
trail, and treats the trail as a closed loop once the path's start/end
points come back within ~40px of each other and the loop spans at least
~60px. A "fist" pose held for ~0.6s while the portal is open closes it.
Tune the thresholds at the top of `src/portal.py` if it's triggering too
easily/rarely for your camera distance.

## Module 4: Face recognition (known people)

Recognizes a fixed set of enrolled people (e.g. people who gave you their
own photo) from a webcam feed or a static image, robust to appearance
changes like glasses or facial hair by storing multiple reference
embeddings per person. Uses InsightFace (`buffalo_l`), not MediaPipe --
MediaPipe has no strong identity-embedding model, only landmarks/detection.

### 1. Add reference photos

```
data/known_faces/
  alice/
    photo1.jpg
    photo2.jpg   # e.g. with glasses
  bob/
    photo1.jpg
```

2-3 photos per person, ideally covering different looks (glasses on/off,
beard grown/shaved), improves match robustness -- the matcher takes the
best score across all of a person's stored embeddings.

### 2. Build the gallery

```
python src/enroll.py
```

Writes `gallery/embeddings.pkl`. First run downloads the `buffalo_l`
model (~350MB) to `~/.insightface`.

### 3. Recognize

Webcam (press `q` to quit):

```
python src/recognize.py
```

Static image:

```
python src/recognize.py --image path/to/photo.jpg
```

Tuning: `MATCH_THRESHOLD` in `src/recognize.py` (default `0.38`) is the
cosine-similarity cutoff for a match vs. "Unknown". Raise it if two
different people are getting confused, lower it if a known person keeps
showing up as Unknown.

## Setup (shared)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
# Computer-Vision-AR
