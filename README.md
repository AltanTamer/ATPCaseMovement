Camera Movement Detection

This project implements a robust camera movement detection system using feature matching and homography analysis. The goal is to accurately detect significant camera movements (such as pan, tilt, or shake) in videos or animated GIFs, while distinguishing these from simple object motion within the scene. The system leverages ORB feature detection and OpenCV's homography estimation to analyze global frame transformations.

## Approach & Movement Detection Logic

- **Feature Extraction:**  
  For each frame, I extract unique feature points using the ORB (Oriented FAST and Rotated BRIEF) algorithm. These features are robust to changes in scale and rotation, and are used for matching and tracking movement across frames.

- **Feature Matching:**  
  For every pair of consecutive frames, I match the extracted features using their binary ORB descriptors. The matching process relies on the Hamming distance, which efficiently compares the binary descriptors and identifies the best correspondences between frames.

- **Homography Estimation:**  
  Once enough matches are found, I estimate a homography matrix using RANSAC. This matrix models the global geometric transformation (translation, rotation, scaling) between the two frames.

- **Movement Scoring:**  
  I analyze the components of the homography matrix to compute a movement score for each frame. This score is a weighted combination of translation, rotation, and scale changes.

- **Movement Classification:**  
  If the movement score exceeds a fixed threshold, the frame is classified as containing significant camera movement. Otherwise, it is considered static or only contains object movement.

This pipeline ensures that only global camera movements (such as pan, tilt, or shake) are detected, while local object motion within the scene does not trigger false positives.

## Challenges & Assumptions

- **Featureless or Repetitive Scenes:**  
  In scenes with very few or repetitive features (e.g., blank walls), homography estimation may be unreliable, potentially leading to missed detections or false positives.

- **Fast or Blurry Motion:**  
  Rapid camera movement or motion blur can reduce feature matching quality, which may affect detection accuracy.

- **Assumptions:**  
  - The input is either a video file (MP4, MOV, AVI, MKV) or an animated GIF.  
  - Single images or static image sequences are not supported, as movement detection requires multiple frames.

## How to Run the App Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AltanTamer/ATPCaseMovement
   cd ATPCaseMovement
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Usage:**
   - Upload a video file (MP4, MOV, AVI, MKV) or an animated GIF.
   - The app will analyze the frames and display detected camera movements, a movement score chart, and a detailed analysis table.

## Live App

https://altanatp.streamlit.app/

![image](https://github.com/user-attachments/assets/a40b9538-5739-40b4-84c8-00f902ba8c1f)
![image](https://github.com/user-attachments/assets/c230e8bb-dc6a-43d7-a0d9-f213967571c8)

