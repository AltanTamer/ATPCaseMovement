# app.py

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import plotly.graph_objects as go
from movement_detector import detect_significant_movement

def load_frames_from_images(uploaded_images):
    """
    Convert uploaded images to RGB numpy arrays.
    """
    frames = []
    for f in uploaded_images:
        img = Image.open(f).convert("RGB")
        frames.append(np.array(img))
    return frames

def load_frames_from_video(uploaded_video):
    """
    Extract frames from uploaded video file.
    """
    # Create temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    # Clean up temporary file
    os.unlink(tfile.name)
    return frames

def load_frames_from_gif(uploaded_gif):
    """
    Extract frames from uploaded GIF file.
    """
    frames = []
    gif = Image.open(uploaded_gif)
    
    try:
        while True:
            # Convert to RGB (in case of RGBA or other formats)
            frame_rgb = gif.convert('RGB')
            # Convert PIL image to numpy array
            frame_np = np.array(frame_rgb)
            frames.append(frame_np)
            # Move to next frame
            gif.seek(gif.tell() + 1)
    except EOFError:
        # End of GIF reached
        pass
    
    return frames

def create_movement_chart(movement_scores, movement_frames):
    """
    Create an interactive chart showing movement scores over time.
    """
    fig = go.Figure()
    
    # Add movement scores line
    fig.add_trace(go.Scatter(
        y=movement_scores,
        mode='lines+markers',
        name='Movement Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Highlight movement frames
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
    
    st.title("📹 Camera Movement Detection")
    st.markdown("""
    This application detects significant camera movement (tilt, pan, translation) in video sequences, GIF files, or image sequences.
    It uses ORB feature matching and homography analysis to accurately distinguish between camera movement and object movement within the scene.
    """)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Threshold setting
    threshold = st.sidebar.slider(
        "Sensitivity Threshold",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Higher values = more sensitive to movement"
    )
    
    # Advanced settings
    min_features = st.sidebar.slider(
        "Minimum Features",
        min_value=5,
        max_value=50,
        value=10,
        help="Minimum number of feature matches required"
    )
    ransac_threshold = st.sidebar.slider(
        "RANSAC Threshold",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Threshold for homography computation"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Media")
        upload_mode = st.radio(
            "Select upload type:",
            ["Video File", "Image Sequence"],
            help="Upload a video file (including GIF) or multiple images"
        )
        
        frames = []
        if upload_mode == "Video File":
            uploaded_video = st.file_uploader(
                "Choose a video file or GIF",
                type=["mp4", "mov", "avi", "mkv", "gif"],
                help="Supported formats: MP4, MOV, AVI, MKV, GIF"
            )
            if uploaded_video is not None:
                with st.spinner("Processing video..."):
                    if uploaded_video.name.lower().endswith('.gif'):
                        frames = load_frames_from_gif(uploaded_video)
                    else:
                        frames = load_frames_from_video(uploaded_video)
        else:
            uploaded_gifs = st.file_uploader(
                "Choose a GIF file (image sequence)",
                type=["gif"],
                accept_multiple_files=False,
                help="Upload a single animated GIF file for image sequence analysis"
            )
            frames = []
            if uploaded_gifs:
                if uploaded_gifs.name.lower().endswith('.gif'):
                    with st.spinner("Processing GIF..."):
                        frames = load_frames_from_gif(uploaded_gifs)
                else:
                    st.warning("Please upload a single animated GIF file for image sequence analysis.")
    
    with col2:
        st.header("📊 Results")
        
        if frames:
            st.success(f"✅ Loaded {len(frames)} frames")
            
            # Run detection
            with st.spinner("Analyzing movement..."):
                results = detect_significant_movement(
                    frames, 
                    threshold=threshold,
                    min_features=min_features,
                    ransac_threshold=ransac_threshold
                )
                movement_frames = results['movement_frames']
                movement_scores = results['movement_scores']
                transformation_data = results['transformation_data']
            
            # Display results
            if movement_frames:
                st.warning(f"🚨 Detected movement in {len(movement_frames)} frames")
                st.write("**Movement detected at frames:**", movement_frames)
            else:
                st.info("✅ No significant camera movement detected")
            
            # Show movement chart
            if len(movement_scores) > 1:
                st.subheader("📈 Movement Analysis Chart")
                chart = create_movement_chart(movement_scores, movement_frames)
                st.plotly_chart(chart, use_container_width=True)
            
            # Show detailed analysis for advanced method
            if transformation_data:
                st.subheader("🔍 Detailed Analysis")
                
                # Create a table of transformation data
                if transformation_data:
                    analysis_data = []
                    for data in transformation_data:
                        analysis_data.append({
                            "Frame": data['frame_idx'],
                            "Matches": data['matches'],
                            "Inliers": data['inliers'],
                            "Score": f"{data['score']:.2f}"
                        })
                    
                    st.dataframe(analysis_data, use_container_width=True)
            
            # Show detected frames
            if movement_frames:
                st.subheader("🎬 Detected Movement Frames")
                
                # Limit the number of frames shown to avoid overwhelming the UI
                max_frames_to_show = 6
                frames_to_show = movement_frames[:max_frames_to_show]
                
                cols = st.columns(min(3, len(frames_to_show)))
                for i, frame_idx in enumerate(frames_to_show):
                    if frame_idx < len(frames):
                        with cols[i % 3]:
                            st.image(
                                frames[frame_idx],
                                caption=f"Frame {frame_idx}",
                                use_column_width=True
                            )
                
                if len(movement_frames) > max_frames_to_show:
                    st.info(f"Showing first {max_frames_to_show} frames. Total: {len(movement_frames)} frames with movement.")
        else:
            st.info("👆 Please upload a video or image sequence to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    - **ORB Feature Detection**: Detects keypoints and descriptors in each frame
    - **Feature Matching**: Matches features between consecutive frames using Hamming distance
    - **Homography Analysis**: Computes transformation matrices to analyze camera movement patterns
    - **Movement Classification**: Distinguishes between translation, rotation, and scaling movements
    
    **Tips:**
    - For best results, use videos with clear, well-lit scenes and good texture
    - Adjust the sensitivity threshold based on your needs (higher = more sensitive)
    - Increase minimum features for more accurate detection
    - This method accurately distinguishes camera movement from object movement
    """)

if __name__ == "__main__":
    main()
