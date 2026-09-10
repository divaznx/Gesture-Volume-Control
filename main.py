import cv2 
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands =1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()

    if not success:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x*frame.shape[1])
                y = int(landmark.y*frame.shape[0])

                cv2.circle(
                    frame,
                    (x,y),
                    5,
                    (0,255,0),
                    -1

                )

    cv2.imshow("Gesture volume control",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()