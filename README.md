<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />

# NPC Detector™ 🎯

## Basic Details
### Team Name: NPC Detector

### Team Members
- Team Lead: Christo Binu - SCMS School of Engineering & Technology

### Project Description
NPC Detector™ is a webcam-based computer vision system that determines how "NPC-like" a person is. It analyzes head movement, blinking, and reaction to a smile challenge to generate a completely unnecessary NPC score.

### The Problem (that doesn't exist)
Have you ever wondered whether someone is secretly an NPC from a video game?

This extremely important question currently has no reliable answer.

### The Solution (that nobody asked for)
NPC Detector™ uses a webcam and computer vision to analyze facial behaviour and movement. It combines head movement, blink frequency, and reaction speed to calculate an NPC score and classify the user as a Background Villager, Shopkeeper NPC, Quest NPC, Playable Character, or Chaotic Player.

It is completely unnecessary and scientifically meaningless. That's the point.

## Technical Details
### Technologies/Components Used
For Software:
- Python
- OpenCV
- MediaPipe
- NumPy
- MediaPipe Face Landmarker
- VS Code
- Git & GitHub

For Hardware:
- Laptop/PC
- Built-in webcam
- No additional hardware required

### Implementation
For Software:

The webcam captures live video and MediaPipe detects facial landmarks.

The system analyzes:
- Head movement
- Eye Aspect Ratio (EAR) for blink detection
- Smile/reaction response
- Behaviour scores

These values are combined to generate the final NPC score and classification.

# Installation

```bash
pip install opencv-python mediapipe numpy