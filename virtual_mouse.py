
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "hand_landmarker.task"

screen_width, screen_height = pyautogui.size()

# Cursor smoothing
SMOOTHING = 5

# Click settings
CLICK_COOLDOWN = 0.45
last_click_time = 0

# Gesture sensitivity
LEFT_CLICK_THRESHOLD = 0.045
RIGHT_CLICK_THRESHOLD = 0.055

# Scrolling settings
SCROLL_SENSITIVITY = 2
SCROLL_MOVEMENT_THRESHOLD = 0.012

# Cursor coordinates
prev_x = 0
prev_y = 0

# Scroll tracking
prev_scroll_y = None

# Track the previous gesture to prevent
# repeated clicks while holding a gesture
previous_gesture = "none"

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def is_finger_extended(hand, tip, pip):
    """
    For index, middle, ring and pinky fingers,
    the fingertip should be above the PIP joint
    in the webcam image when the finger is raised.
    """
    return hand[tip].y < hand[pip].y


def is_fist(hand):
    """
    Detect a fist by checking whether all four
    non-thumb fingers are folded.
    """
    index_folded = hand[8].y > hand[6].y
    middle_folded = hand[12].y > hand[10].y
    ring_folded = hand[16].y > hand[14].y
    pinky_folded = hand[20].y > hand[18].y

    return (
        index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    )


def show_status(frame, message, color=(0, 255, 0)):
    cv2.putText(
        frame,
        message,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


# ==========================================
# 4. WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam.")
    raise SystemExit

try:
    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Video timestamps must increase
            timestamp = time.monotonic_ns() // 1_000_000

            # Detect hand
            results = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            gesture = "none"
            status = "NO HAND DETECTED"

            if results.hand_landmarks:

                hand = results.hand_landmarks[0]

                # Finger landmarks
                thumb = hand[4]
                index = hand[8]
                middle = hand[12]

                ring = hand[16]
                pinky = hand[20]

                # ==================================
                # 5. CURSOR MOVEMENT
                # ==================================

                # ONLY index fingertip controls cursor
                screen_x = int(
                    index.x * screen_width
                )

                screen_y = int(
                    index.y * screen_height
                )

                # Smooth cursor movement
                curr_x = prev_x + (
                    screen_x - prev_x
                ) / SMOOTHING

                curr_y = prev_y + (
                    screen_y - prev_y
                ) / SMOOTHING

                # Keep coordinates within screen
                curr_x = max(
                    0, min(screen_width - 1, curr_x)
                )

                curr_y = max(
                    0, min(screen_height - 1, curr_y)
                )

                pyautogui.moveTo(
                    int(curr_x),
                    int(curr_y)
                )

                prev_x = curr_x
                prev_y = curr_y

                # ==================================
                # 6. FINGER STATES
                # ==================================

                index_up = is_finger_extended(
                    hand, 8, 6
                )

                middle_up = is_finger_extended(
                    hand, 12, 10
                )

                ring_up = is_finger_extended(
                    hand, 16, 14
                )

                pinky_up = is_finger_extended(
                    hand, 20, 18
                )

                # Distance between fingertips
                thumb_index = distance(
                    thumb, index
                )

                index_middle = distance(
                    index, middle
                )

                # ==================================
                # 7. GESTURE RECOGNITION
                # ==================================

                fist = is_fist(hand)

                # Two-finger scroll pose:
                # Index and middle raised;
                # ring and pinky folded.
                scroll_pose = (
                    index_up
                    and middle_up
                    and not ring_up
                    and not pinky_up
                    and index_middle > 0.06
                    and thumb_index > 0.07
                )

                # Fist takes priority over other gestures
                if fist:
                    gesture = "fist"

                # Thumb + index touch = left click
                elif (
                    thumb_index < LEFT_CLICK_THRESHOLD
                ):
                    gesture = "left_click"

                # Index + middle touch = right click
                elif (
                    index_middle < RIGHT_CLICK_THRESHOLD
                    and thumb_index > 0.07
                ):
                    gesture = "right_click"

                # Two raised fingers = scrolling
                elif scroll_pose:
                    gesture = "scroll"

                else:
                    gesture = "none"

                # ==================================
                # 8. CLICK CONTROLS
                # ==================================

                current_time = time.time()

                # Trigger clicks only when a new
                # gesture is detected, not every frame.
                new_gesture = (
                    gesture != previous_gesture
                )

                can_click = (
                    current_time - last_click_time
                    >= CLICK_COOLDOWN
                )

                if new_gesture and can_click:

                    if gesture == "left_click":

                        pyautogui.click()
                        last_click_time = current_time

                        status = "LEFT CLICK"

                    elif gesture == "right_click":

                        pyautogui.rightClick()
                        last_click_time = current_time

                        status = "RIGHT CLICK"

                    elif gesture == "fist":

                        pyautogui.doubleClick(
                            interval=0.12
                        )

                        last_click_time = current_time

                        status = "DOUBLE CLICK"

                # Display current action
                if gesture == "scroll":
                    status = "SCROLL MODE"

                elif gesture == "none":
                    status = "INDEX FINGER: MOVE"

                elif gesture == "left_click":
                    if status != "LEFT CLICK":
                        status = "LEFT CLICK READY"

                elif gesture == "right_click":
                    if status != "RIGHT CLICK":
                        status = "RIGHT CLICK READY"

                elif gesture == "fist":
                    if status != "DOUBLE CLICK":
                        status = "FIST DETECTED"

                # ==================================
                # 9. SCROLLING
                # ==================================

                if scroll_pose:

                    # Average vertical position of
                    # index and middle fingertips
                    current_scroll_y = (
                        index.y + middle.y
                    ) / 2

                    if prev_scroll_y is not None:

                        movement = (
                            prev_scroll_y
                            - current_scroll_y
                        )

                        if (
                            abs(movement)
                            >= SCROLL_MOVEMENT_THRESHOLD
                        ):

                            scroll_amount = max(
                                1,
                                int(
                                    abs(movement)
                                    * 100
                                    * SCROLL_SENSITIVITY
                                )
                            )

                            # Hand moves up = scroll up
                            # Hand moves down = scroll down
                            if movement > 0:
                                pyautogui.scroll(
                                    scroll_amount
                                )
                            else:
                                pyautogui.scroll(
                                    -scroll_amount
                                )

                    prev_scroll_y = current_scroll_y

                else:
                    # Reset baseline when leaving
                    # scroll mode to prevent jumps.
                    prev_scroll_y = None

                # ==================================
                # 10. DRAW LANDMARKS
                # ==================================

                for point in hand:

                    x = int(
                        point.x * frame.shape[1]
                    )

                    y = int(
                        point.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Show index fingertip distinctly
                index_x = int(
                    index.x * frame.shape[1]
                )

                index_y = int(
                    index.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (index_x, index_y),
                    9,
                    (0, 0, 255),
                    2
                )

                previous_gesture = gesture

            else:
                # Reset gesture and scroll tracking
                previous_gesture = "none"
                prev_scroll_y = None

            # ==================================
            # 11. DISPLAY
            # ==================================

            cv2.putText(
                frame,
                "VIRTUAL MOUSE",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            show_status(frame, status)

            cv2.putText(
                frame,
                "Move: Index | L: Thumb+Index",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "R: Index+Middle | Scroll: 2 fingers",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "Double click: Fist | Quit: Q",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Virtual Mouse",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "hand_landmarker.task"

screen_width, screen_height = pyautogui.size()

# Cursor smoothing
SMOOTHING = 5

# Click settings
CLICK_COOLDOWN = 0.45
last_click_time = 0

# Gesture sensitivity
LEFT_CLICK_THRESHOLD = 0.045
RIGHT_CLICK_THRESHOLD = 0.055

# Scrolling settings
SCROLL_SENSITIVITY = 2
SCROLL_MOVEMENT_THRESHOLD = 0.012

# Cursor coordinates
prev_x = 0
prev_y = 0

# Scroll tracking
prev_scroll_y = None

# Track the previous gesture to prevent
# repeated clicks while holding a gesture
previous_gesture = "none"

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def is_finger_extended(hand, tip, pip):
    """
    For index, middle, ring and pinky fingers,
    the fingertip should be above the PIP joint
    in the webcam image when the finger is raised.
    """
    return hand[tip].y < hand[pip].y


def is_fist(hand):
    """
    Detect a fist by checking whether all four
    non-thumb fingers are folded.
    """
    index_folded = hand[8].y > hand[6].y
    middle_folded = hand[12].y > hand[10].y
    ring_folded = hand[16].y > hand[14].y
    pinky_folded = hand[20].y > hand[18].y

    return (
        index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    )


def show_status(frame, message, color=(0, 255, 0)):
    cv2.putText(
        frame,
        message,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


# ==========================================
# 4. WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam.")
    raise SystemExit

try:
    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Video timestamps must increase
            timestamp = time.monotonic_ns() // 1_000_000

            # Detect hand
            results = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            gesture = "none"
            status = "NO HAND DETECTED"

            if results.hand_landmarks:

                hand = results.hand_landmarks[0]

                # Finger landmarks
                thumb = hand[4]
                index = hand[8]
                middle = hand[12]

                ring = hand[16]
                pinky = hand[20]

                # ==================================
                # 5. CURSOR MOVEMENT
                # ==================================

                # ONLY index fingertip controls cursor
                screen_x = int(
                    index.x * screen_width
                )

                screen_y = int(
                    index.y * screen_height
                )

                # Smooth cursor movement
                curr_x = prev_x + (
                    screen_x - prev_x
                ) / SMOOTHING

                curr_y = prev_y + (
                    screen_y - prev_y
                ) / SMOOTHING

                # Keep coordinates within screen
                curr_x = max(
                    0, min(screen_width - 1, curr_x)
                )

                curr_y = max(
                    0, min(screen_height - 1, curr_y)
                )

                pyautogui.moveTo(
                    int(curr_x),
                    int(curr_y)
                )

                prev_x = curr_x
                prev_y = curr_y

                # ==================================
                # 6. FINGER STATES
                # ==================================

                index_up = is_finger_extended(
                    hand, 8, 6
                )

                middle_up = is_finger_extended(
                    hand, 12, 10
                )

                ring_up = is_finger_extended(
                    hand, 16, 14
                )

                pinky_up = is_finger_extended(
                    hand, 20, 18
                )

                # Distance between fingertips
                thumb_index = distance(
                    thumb, index
                )

                index_middle = distance(
                    index, middle
                )

                # ==================================
                # 7. GESTURE RECOGNITION
                # ==================================

                fist = is_fist(hand)

                # Two-finger scroll pose:
                # Index and middle raised;
                # ring and pinky folded.
                scroll_pose = (
                    index_up
                    and middle_up
                    and not ring_up
                    and not pinky_up
                    and index_middle > 0.06
                    and thumb_index > 0.07
                )

                # Fist takes priority over other gestures
                if fist:
                    gesture = "fist"

                # Thumb + index touch = left click
                elif (
                    thumb_index < LEFT_CLICK_THRESHOLD
                ):
                    gesture = "left_click"

                # Index + middle touch = right click
                elif (
                    index_middle < RIGHT_CLICK_THRESHOLD
                    and thumb_index > 0.07
                ):
                    gesture = "right_click"

                # Two raised fingers = scrolling
                elif scroll_pose:
                    gesture = "scroll"

                else:
                    gesture = "none"

                # ==================================
                # 8. CLICK CONTROLS
                # ==================================

                current_time = time.time()

                # Trigger clicks only when a new
                # gesture is detected, not every frame.
                new_gesture = (
                    gesture != previous_gesture
                )

                can_click = (
                    current_time - last_click_time
                    >= CLICK_COOLDOWN
                )

                if new_gesture and can_click:

                    if gesture == "left_click":

                        pyautogui.click()
                        last_click_time = current_time

                        status = "LEFT CLICK"

                    elif gesture == "right_click":

                        pyautogui.rightClick()
                        last_click_time = current_time

                        status = "RIGHT CLICK"

                    elif gesture == "fist":

                        pyautogui.doubleClick(
                            interval=0.12
                        )

                        last_click_time = current_time

                        status = "DOUBLE CLICK"

                # Display current action
                if gesture == "scroll":
                    status = "SCROLL MODE"

                elif gesture == "none":
                    status = "INDEX FINGER: MOVE"

                elif gesture == "left_click":
                    if status != "LEFT CLICK":
                        status = "LEFT CLICK READY"

                elif gesture == "right_click":
                    if status != "RIGHT CLICK":
                        status = "RIGHT CLICK READY"

                elif gesture == "fist":
                    if status != "DOUBLE CLICK":
                        status = "FIST DETECTED"

                # ==================================
                # 9. SCROLLING
                # ==================================

                if scroll_pose:

                    # Average vertical position of
                    # index and middle fingertips
                    current_scroll_y = (
                        index.y + middle.y
                    ) / 2

                    if prev_scroll_y is not None:

                        movement = (
                            prev_scroll_y
                            - current_scroll_y
                        )

                        if (
                            abs(movement)
                            >= SCROLL_MOVEMENT_THRESHOLD
                        ):

                            scroll_amount = max(
                                1,
                                int(
                                    abs(movement)
                                    * 100
                                    * SCROLL_SENSITIVITY
                                )
                            )

                            # Hand moves up = scroll up
                            # Hand moves down = scroll down
                            if movement > 0:
                                pyautogui.scroll(
                                    scroll_amount
                                )
                            else:
                                pyautogui.scroll(
                                    -scroll_amount
                                )

                    prev_scroll_y = current_scroll_y

                else:
                    # Reset baseline when leaving
                    # scroll mode to prevent jumps.
                    prev_scroll_y = None

                # ==================================
                # 10. DRAW LANDMARKS
                # ==================================

                for point in hand:

                    x = int(
                        point.x * frame.shape[1]
                    )

                    y = int(
                        point.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Show index fingertip distinctly
                index_x = int(
                    index.x * frame.shape[1]
                )

                index_y = int(
                    index.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (index_x, index_y),
                    9,
                    (0, 0, 255),
                    2
                )

                previous_gesture = gesture

            else:
                # Reset gesture and scroll tracking
                previous_gesture = "none"
                prev_scroll_y = None

            # ==================================
            # 11. DISPLAY
            # ==================================

            cv2.putText(
                frame,
                "VIRTUAL MOUSE",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            show_status(frame, status)

            cv2.putText(
                frame,
                "Move: Index | L: Thumb+Index",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "R: Index+Middle | Scroll: 2 fingers",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "Double click: Fist | Quit: Q",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "Virtual Mouse",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
