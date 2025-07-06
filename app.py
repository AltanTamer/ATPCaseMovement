import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import plotly.graph_objects as go
from movement_detector import detect_significant_movement

def load_frames_from_video(uploaded_video):

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    os.unlink(tfile.name)
    return frames

def load_frames_from_gif(uploaded_gif):
    frames = []
    gif = Image.open(uploaded_gif)
    
    try:
        while True:
            frame_rgb = gif.convert('RGB')
            frame_np = np.array(frame_rgb)
            frames.append(frame_np)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    return frames

def create_movement_chart(movement_scores, movement_frames):

    fig = go.Figure()
    

    fig.add_trace(go.Scatter(
        y=movement_scores,
        mode='lines+markers',
        name='Movement Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    

    if movement_frames:
        movement_scores_highlight = [movement_scores[i] if i < len(movement_scores) else 0 for i in movement_frames]
        fig.add_trace(go.Scatter(
            x=movement_frames,
            y=movement_scores_highlight,
            mode='markers',
            name='Detected Movement',
            marker=dict(color='red', size=10, symbol='x'),
            hovertemplate='Frame %{x}<br>Score: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Camera Movement Detection Results',
        xaxis_title='Frame Index',
        yaxis_title='Movement Score',
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Camera Movement Detection",
        page_icon="📹",
        layout="wide"
    )
    
    left, center, right = st.columns([1, 3, 1])
    with center:
        st.title("Camera Movement Detection")
        st.markdown("""
        This application detects significant camera movement (tilt, pan, translation) in video sequences, GIF files, or image sequences.
        It uses ORB feature matching and homography analysis to accurately distinguish between camera movement and object movement within the scene.
        """)
        st.header("Upload Media")
        uploaded_file = st.file_uploader(
            "Choose a video file or GIF",
            type=["mp4", "mov", "avi", "mkv", "gif"],
            accept_multiple_files=False,
            help="Upload a video (MP4, MOV, AVI, MKV) or an animated GIF file"
        )
        frames = []
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                if uploaded_file.name.lower().endswith('.gif'):
                    frames = load_frames_from_gif(uploaded_file)
                else:
                    frames = load_frames_from_video(uploaded_file)

        st.header("Results")
        if frames:
            st.success(f"Loaded {len(frames)} frames")
            with st.spinner("Analyzing movement..."):
                results = detect_significant_movement(
                    frames, 
                    threshold=50,
                    min_features=10,
                    ransac_threshold=3.0
                )
                movement_frames = results['movement_frames']
                movement_scores = results['movement_scores']
                transformation_data = results['transformation_data']
            if movement_frames:
                st.warning(f"Detected movement in {len(movement_frames)} frames")
                st.write("**Movement detected at frames:**", movement_frames)
            else:
                st.info("No significant camera movement detected")
            if len(movement_scores) > 1:
                st.subheader("Movement Analysis Chart")
                with st.container():
                    chart = create_movement_chart(movement_scores, movement_frames)
                    chart.update_layout(height=500, width=None)
                    st.plotly_chart(chart, use_container_width=True)
            if transformation_data:
                st.subheader("Detailed Analysis")
                if transformation_data:
                    analysis_data = []
                    for data in transformation_data:
                        movement_type = "Camera movement" if (data['score'] > 50 and data['inliers'] / max(data['matches'],1) > 0.5) else "Object/static"
                        analysis_data.append({
                            "Frame": data['frame_idx'],
                            "Matches": data['matches'],
                            "Inliers": data['inliers'],
                            "Score": f"{data['score']:.2f}",
                            "Movement Type": movement_type
                        })
                    st.dataframe(analysis_data, use_container_width=True)
            if movement_frames:
                st.subheader("Detected Movement Frames")
                max_frames_to_show = 6
                frames_to_show = movement_frames[:max_frames_to_show]
                cols = st.columns(min(3, len(frames_to_show)))
                for i, frame_idx in enumerate(frames_to_show):
                    if frame_idx < len(frames):
                        with cols[i % 3]:
                            st.image(
                                frames[frame_idx],
                                caption=f"Frame {frame_idx}",
                                use_container_width=True
                            )
                if len(movement_frames) > max_frames_to_show:
                    st.info(f"Showing first {max_frames_to_show} frames. Total: {len(movement_frames)} frames with movement.")
        else:
            st.info("Please upload a video or image sequence to begin analysis")
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    - **ORB Feature Detection**: Detects keypoints and descriptors in each frame
    - **Feature Matching**: Matches features between consecutive frames using Hamming distance
    - **Homography Analysis**: Computes transformation matrices to analyze camera movement patterns
    - **Movement Classification**: Distinguishes between translation, rotation, and scaling movements
    """)

if __name__ == "__main__":
    main()
