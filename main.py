import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pycaw.pycaw import AudioUtilities


# -----------------------------
# MediaPipe setup
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Windows volume setup
# -----------------------------

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

volume_range = volume.GetVolumeRange()

min_volume = volume_range[0]
max_volume = volume_range[1]


# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)


    # -----------------------------
    # Hand detected
    # -----------------------------

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            # Landmark 4 = thumb tip
            # Landmark 8 = index finger tip

            thumb = hand[4]
            index = hand[8]

            thumb_x = int(
                thumb.x * frame.shape[1]
            )
            thumb_y = int(
                thumb.y * frame.shape[0]
            )

            index_x = int(
                index.x * frame.shape[1]
            )
            index_y = int(
                index.y * frame.shape[0]
            )


            # -----------------------------
            # Calculate distance
            # -----------------------------

            distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )


            # -----------------------------
            # Map distance → volume %
            # -----------------------------

            min_distance = 30
            max_distance = 200

            distance = max(
                min_distance,
                min(max_distance, distance)
            )

            volume_percent = (
                (distance - min_distance)
                / (max_distance - min_distance)
            ) * 100


            # -----------------------------
            # Map volume % → Windows dB
            # -----------------------------

            volume_db = (
                min_volume
                + (volume_percent / 100)
                * (max_volume - min_volume)
            )

            volume.SetMasterVolumeLevel(
                volume_db,
                None
            )


            # -----------------------------
            # Draw thumb
            # -----------------------------

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (0, 255, 0),
                -1
            )


            # -----------------------------
            # Draw index finger
            # -----------------------------

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                -1
            )


            # -----------------------------
            # Draw line
            # -----------------------------

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 0, 0),
                3
            )


            # -----------------------------
            # Display distance
            # -----------------------------

            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


            # -----------------------------
            # Display volume
            # -----------------------------

            cv2.putText(
                frame,
                f"Volume: {int(volume_percent)}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


    # -----------------------------
    # Show frame
    # -----------------------------

    cv2.imshow(
        "Gesture volume control",
        frame
    )


    # Close window
    if cv2.getWindowProperty(
        "Gesture volume control",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()

detector.close()