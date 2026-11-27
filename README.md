Virtual Mouse Using Hand Gestures

 Project Overview
Virtual Mouse is a Python-based application that allows users to control their computer's mouse using hand gestures through a webcam. It uses computer vision and hand landmark detection to track hand movements and perform mouse operations without requiring a physical mouse.

Features
- Cursor Movement: Move the mouse cursor using hand movements.
- Click Detection: Perform mouse clicks using specific hand gestures.
- Real-Time Hand Tracking: Detect and track hand landmarks using a webcam.
- Touchless Interaction: Control the computer without physically touching a mouse.

Technologies Used
- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

 Project Structure

```text
Virtual-Mouse/
│
├── virtual_mouse.py
├── hand_landmarker.task
└── README.md
```

 Installation & Setup

1. Clone the repository

```bash
git clone https://github.com/aryashamka20105-cmd/Virtual-Mouse.git
```

2. Navigate to the project directory

```bash
cd Virtual-Mouse
```

3. Install the required dependencies

```bash
pip install opencv-python mediapipe pyautogui numpy
```

4. Run the application

```bash
python virtual_mouse.py
```

 Requirements
- Python installed on your system.
- A working webcam.
- Required Python libraries installed.
- `hand_landmarker.task` model file in the project directory.

Applications
- Touchless computer interaction.
- Accessibility support.
- Human-computer interaction.
- Gesture-based control systems.

 Future Improvements
- Add additional hand gestures for scrolling and dragging.
- Improve gesture recognition accuracy.
- Add support for multiple hand gestures and custom controls.
- Optimize performance for smoother cursor movement.

👩‍💻 Author
Arya Shamkha

GitHub: [@aryashamka20105-cmd](https://github.com/aryashamka20105-cmd)

---

⭐ If you find this project interesting, consider giving it a star!
