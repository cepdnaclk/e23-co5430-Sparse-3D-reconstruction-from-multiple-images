"""Feature detection and pairwise matching."""

import os

import cv2

from .models import Frame
from .reporting import RUN_STATS, record_match_count


def make_detector(feature_type, max_features=4000):
    feature_type = feature_type.lower()
    if feature_type == "sift":
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    if feature_type == "orb":
        return (
            cv2.ORB_create(nfeatures=max_features, fastThreshold=7),
            cv2.NORM_HAMMING,
        )
    raise ValueError(f"Unknown feature_type '{feature_type}', use 'sift' or 'orb'")


def detect_features(paths, feature_type="sift", max_features=4000):
    detector, _ = make_detector(feature_type, max_features)
    frames = []
    for index, path in enumerate(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image {path}")
        keypoints, descriptors = detector.detectAndCompute(image, None)
        frames.append(Frame(index, path, keypoints, descriptors, image.shape))
        RUN_STATS["n_keypoints"][index] = len(keypoints)
        print(
            f"[features:{feature_type}] {os.path.basename(path)}: "
            f"{len(keypoints)} keypoints"
        )
    return frames


def match_pair(f1, f2, ratio=0.75, norm_type=cv2.NORM_L2):
    if f1.desc is None or f2.desc is None or len(f1.desc) < 8 or len(f2.desc) < 8:
        return []
    matcher = cv2.BFMatcher(norm_type)
    pairs = matcher.knnMatch(f1.desc, f2.desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        match, neighbor = pair
        if match.distance < ratio * neighbor.distance:
            good.append(match)
    return good


def match_all(frames, norm_type, matching_strategy="exhaustive", match_window=6):
    matches_table = {}
    n_frames = len(frames)
    if matching_strategy == "sequential":
        pairs = set()
        for i in range(n_frames):
            for distance in range(1, match_window + 1):
                j = (i + distance) % n_frames
                pairs.add((min(i, j), max(i, j)))
        print(
            f"[match] sequential strategy: {len(pairs)} pairs "
            f"(vs {n_frames * (n_frames - 1) // 2} for exhaustive)"
        )
        for i, j in sorted(pairs):
            matches = match_pair(frames[i], frames[j], norm_type=norm_type)
            record_match_count(i, j, len(matches))
            if len(matches) >= 20:
                matches_table[(i, j)] = matches
    else:
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                matches = match_pair(frames[i], frames[j], norm_type=norm_type)
                record_match_count(i, j, len(matches))
                if len(matches) >= 20:
                    matches_table[(i, j)] = matches
            print(f"[match] frame {i} vs rest done")
    return matches_table
