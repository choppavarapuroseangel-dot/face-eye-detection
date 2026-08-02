# Real-Time Face & Eye Detection

Real-time face and eye detection from a webcam feed, using OpenCV's Haar
Cascade classifiers. Built during a software development internship at
Innovate Intern (Chennai) — applying classical AI/ML computer-vision
techniques to detect faces and eyes live, the kind of pipeline used in
security systems, facial recognition, and human-computer interaction.

## How it works

1. Each frame is converted to grayscale and histogram-equalized, which
   makes Haar features more consistent under uneven lighting.
2. `haarcascade_frontalface_default.xml` finds face regions in the frame.
3. `haarcascade_eye.xml` searches only *inside* each detected face for
   eyes — scanning the whole frame for eyes produces far more false
   positives than constraining the search to a known face region.
4. Detections below face regions (nostrils, mouth corners the eye
   classifier sometimes misfires on) are filtered by position.
5. Matches are drawn as bounding boxes and shown live, with an FPS
   counter so you can see the pipeline is keeping up with the camera.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python detect.py                     # live webcam (default index 0)
python detect.py --source 1          # a different camera index
python detect.py --source clip.mp4   # run on a video file instead
python detect.py --source photo.jpg  # run on a single image
python detect.py --list-cameras      # see which camera indexes exist
```

Press **q** or **Esc** to close the live window. Image mode saves the
annotated result next to the input as `<name>_detected.jpg`.

## Stack

Python · OpenCV (`cv2`) · Haar Cascade classifiers

## Tested against

Beyond a live webcam, `detect_and_annotate()` was run against seven still
photos — sourced from OpenCV's own official test corpus plus one personal
photo — to check it generalizes past one face in good lighting:

| Image | Result |
|---|---|
| Single portrait (own photo) | Face found. 1 false positive on a plaid shirt lower in the frame. |
| [`messi5.jpg`](https://github.com/opencv/opencv/blob/master/samples/data/messi5.jpg) — single face at an angle, mid-action | Face found. 1 false positive on a striped jersey/blurred background. |
| `karen-and-rob.png` — two people | Both faces found, no false positives. |
| A seven-person group photo | All 7 faces found, no false positives. |
| `audrybt1.png` — single portrait against heavily patterned wallpaper | Face found, no false positives — texture false-positives aren't universal. |
| `er.png` — six-person cast photo, studio lighting | 5 of 6 real faces found; the 6th (seated, angled) was missed, and a false positive landed on a badge/lanyard instead. |
| `churchill-downs.png` — architectural photo, no people | Correctly 0 detections — confirms it doesn't hallucinate faces on columns, roofline, or signage text. |

Net across all seven: **17 of 18** real faces found, **3** false positives —
all three on repeating fabric/lanyard patterns, none on architecture or
plain backgrounds.

## Notes

- Needs a webcam and, on macOS, camera permission granted to your
  terminal or IDE (System Settings → Privacy & Security → Camera).
- **Haar cascades reliably false-positive on repeating high-frequency
  textures** — confirmed independently on two unrelated test photos, both
  times on patterned clothing (a plaid shirt, a striped jersey). This
  isn't fixable by tuning `minNeighbors` (tested up to 12, no change) or
  by requiring an eye match inside the box before accepting a face — that
  was tried, and rejected 3/3 real faces in testing because small or
  angled faces often have no cleanly-detectable eye, while loosening the
  eye parameters enough to catch those real eyes let the eye cascade match
  the same fabric that fooled the face cascade. It's a genuine limitation
  of the classical approach, not a bug in this script — deep-learning
  detectors (MTCNN, MediaPipe, YuNet) handle this far better, at the cost
  of a much heavier dependency.
- **Requires OpenCV 4.x.** OpenCV 5.0 removed `cv2.CascadeClassifier`
  and no longer bundles the Haar cascade XML files at all, in favor of
  the newer DNN-based `FaceDetectorYN`. `requirements.txt` pins
  `opencv-python<5` for that reason — installing latest will break this
  script.
