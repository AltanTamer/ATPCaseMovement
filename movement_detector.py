import cv2
import numpy as np
from typing import List, Tuple, Dict

def detect_significant_movement(
    frames: List[np.ndarray],
    threshold: float = 50.0,
    min_features: int = 10,
    ransac_threshold: float = 3.0
    
) -> Dict[str, List]:
    movement_frames = []
    movement_scores = []
    transformation_data = []
    orb = cv2.ORB_create(nfeatures=1000)
    prev_gray = None
    prev_keypoints = None
    prev_descriptors = None
    
    
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if (prev_keypoints is not None and prev_descriptors is not None and 
                len(keypoints) > min_features and len(prev_keypoints) > min_features):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(prev_descriptors, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) >= min_features:
                    src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
                    
                    if H is not None:
                        movement_score = analyze_transformation(H, len(matches), len(matches) * np.sum(mask) / len(mask))
                        transformation_data.append({
                            'frame_idx': idx,
                            'homography': H,
                            'matches': len(matches),
                            'inliers': int(np.sum(mask)),
                            'score': movement_score
                        })
                        movement_scores.append(movement_score)
                        
                        if movement_score > threshold:
                            movement_frames.append(idx)
                    
                    else:
                        movement_scores.append(100.0)
                        movement_frames.append(idx)
                        transformation_data.append({
                            'frame_idx': idx,
                            'homography': None,
                            'matches': len(matches),
                            'inliers': 0,
                            'score': 100.0
                        })
                
                else:
                    movement_scores.append(100.0)
                    movement_frames.append(idx)
                    transformation_data.append({
                        'frame_idx': idx,
                        'homography': None,
                        'matches': len(matches),
                        'inliers': 0,
                        'score': 100.0
                    })
            
            else:
                movement_scores.append(100.0)
                movement_frames.append(idx)
                transformation_data.append({
                    'frame_idx': idx,
                    'homography': None,
                    'matches': 0,
                    'inliers': 0,
                    'score': 100.0
                })
        prev_gray = gray
        prev_keypoints, prev_descriptors = orb.detectAndCompute(gray, None)
    
    return {
        'movement_frames': movement_frames,
        'movement_scores': movement_scores,
        'transformation_data': transformation_data
    }

def analyze_transformation(H: np.ndarray, num_matches: int, num_inliers: int) -> float:
    if H is None:
        return 100.0
    tx = H[0, 2]
    ty = H[1, 2]
    a = H[0, 0]
    b = H[0, 1]
    c = H[1, 0]
    d = H[1, 1]
    translation_magnitude = np.sqrt(tx**2 + ty**2)
    rotation_angle = np.arctan2(b, a)
    scale_factor = np.sqrt(a**2 + c**2)
    normalized_translation = min(translation_magnitude / 8.0, 1.0)
    normalized_rotation = min(abs(rotation_angle) / (np.pi/18), 1.0)
    normalized_scale = min(abs(scale_factor - 1.0) / 0.08, 1.0)
    movement_score = (
        0.6 * normalized_translation * 100 +
        0.25 * normalized_rotation * 100 +
        0.15 * normalized_scale * 100
    )
    if num_matches < 15:
        movement_score *= 1.1
    if num_inliers / num_matches < 0.4:
        movement_score *= 1.05
    return movement_score


