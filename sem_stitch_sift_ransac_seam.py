from __future__ import annotations

import argparse
import math
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class FeatureData:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None


@dataclass
class PairMatch:
    i: int
    j: int
    H_i_to_j: np.ndarray
    good_matches: int
    inliers: int
    inlier_ratio: float
    score: float
    center_shift: tuple[float, float]
    scale: float
    rotation_deg: float


@dataclass
class SeamRecord:
    seam: np.ndarray
    layout: str
    color_index: int


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def discover_images(input_dir: Path) -> list[Path]:
    images = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    images.sort(key=natural_key)
    if len(images) < 2:
        raise RuntimeError(
            f"Only {len(images)} input image(s) were found. "
            f"Place at least two images in the {input_dir} folder."
        )
    return images


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to open image: {path}")

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    raise RuntimeError(f"Unsupported image array: {path} / shape={image.shape}")


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray


def robust_to_uint8(gray: np.ndarray) -> np.ndarray:
    finite = np.isfinite(gray)
    if not np.any(finite):
        return np.zeros(gray.shape, dtype=np.uint8)

    values = gray[finite]
    lo = float(np.percentile(values, 0.5))
    hi = float(np.percentile(values, 99.5))
    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.uint8)

    scaled = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def feature_gray(image: np.ndarray) -> np.ndarray:
    gray8 = robust_to_uint8(to_gray_float(image))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray8)


def image_center(shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    return np.array([w / 2.0, h / 2.0], dtype=np.float64)


def homography_geometry(H: np.ndarray, source_shape: tuple[int, ...]) -> tuple[float, float, tuple[float, float]]:
    h, w = source_shape[:2]
    pts = np.float32([[[0, 0], [w, 0], [w, h], [0, h], [w / 2.0, h / 2.0]]])
    transformed = cv2.perspectiveTransform(pts, H)[0].astype(np.float64)

    v_x = transformed[1] - transformed[0]
    v_y = transformed[3] - transformed[0]
    sx = np.linalg.norm(v_x) / max(float(w), 1.0)
    sy = np.linalg.norm(v_y) / max(float(h), 1.0)
    scale = math.sqrt(max(sx * sy, 1.0e-12))
    rotation = math.degrees(math.atan2(v_x[1], v_x[0]))
    center_shift = transformed[4] - image_center(source_shape)
    return scale, rotation, (float(center_shift[0]), float(center_shift[1]))


def estimate_pair(
    image_i: np.ndarray,
    image_j: np.ndarray,
    features_i: FeatureData,
    features_j: FeatureData,
    *,
    i: int,
    j: int,
    ratio_test: float,
    ransac_threshold: float,
    min_good_matches: int,
    min_inliers: int,
    min_inlier_ratio: float,
    min_scale: float,
    max_scale: float,
    max_rotation_deg: float,
) -> PairMatch | None:
    kp_i, des_i = features_i.keypoints, features_i.descriptors
    kp_j, des_j = features_j.keypoints, features_j.descriptors

    if des_i is None or des_j is None or len(kp_i) < 4 or len(kp_j) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des_i, des_j, k=2)

    good: list[cv2.DMatch] = []
    used_ratio = ratio_test
    for candidate_ratio in (ratio_test, 0.82, 0.85, 0.88, 0.90):
        good = [m for m, n in pairs if m.distance < candidate_ratio * n.distance]
        used_ratio = candidate_ratio
        if len(good) >= max(min_good_matches, 20):
            break

    if len(good) < min_good_matches:
        return None

    src = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_j[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        ransac_threshold,
        maxIters=30000,
        confidence=0.999,
    )
    if H is None or inlier_mask is None:
        return None

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / max(len(good), 1)
    if inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return None

    scale, rotation, center_shift = homography_geometry(H, image_i.shape)
    if not (min_scale <= scale <= max_scale):
        return None
    normalized_rotation = ((rotation + 180.0) % 360.0) - 180.0
    if abs(normalized_rotation) > max_rotation_deg:
        return None

    # Score favors many geometrically consistent inliers and penalizes relaxed ratio tests.
    ratio_penalty = max(0.75, 1.0 - max(0.0, used_ratio - ratio_test) * 1.5)
    score = float(inliers) * (0.5 + 0.5 * inlier_ratio) * ratio_penalty

    return PairMatch(
        i=i,
        j=j,
        H_i_to_j=H.astype(np.float64),
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        score=score,
        center_shift=center_shift,
        scale=scale,
        rotation_deg=normalized_rotation,
    )


def compute_features(images: list[np.ndarray], max_features: int) -> list[FeatureData]:
    print("\n[1/6] Computing SIFT features for each image")
    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.008,
        edgeThreshold=20,
        sigma=1.6,
    )
    features: list[FeatureData] = []
    for idx, image in enumerate(images, start=1):
        gray = feature_gray(image)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features.append(FeatureData(keypoints=keypoints, descriptors=descriptors))
        count = 0 if keypoints is None else len(keypoints)
        print(f"  - Image {idx:02d}: {count} keypoints")
    return features


def pairwise_matches(
    images: list[np.ndarray], features: list[FeatureData], args: argparse.Namespace
) -> list[PairMatch]:
    n = len(images)
    total = n * (n - 1) // 2
    results: list[PairMatch] = []
    done = 0

    print(f"\n[2/6] SIFT matching and RANSAC homography estimation for all image pairs ({total} pairs)")
    for i in range(n):
        for j in range(i + 1, n):
            done += 1
            print(f"  - [{done:>4}/{total}] {i+1:02d} <-> {j+1:02d}", end="", flush=True)
            match = estimate_pair(
                images[i],
                images[j],
                features[i],
                features[j],
                i=i,
                j=j,
                ratio_test=args.ratio_test,
                ransac_threshold=args.ransac_threshold,
                min_good_matches=args.min_good_matches,
                min_inliers=args.min_inliers,
                min_inlier_ratio=args.min_inlier_ratio,
                min_scale=args.min_scale,
                max_scale=args.max_scale,
                max_rotation_deg=args.max_rotation,
            )
            if match is None:
                print(" : no connection")
            else:
                results.append(match)
                print(
                    f" : OK / matches={match.good_matches}, "
                    f"inliers={match.inliers}, ratio={match.inlier_ratio:.2f}, "
                    f"score={match.score:.1f}"
                )
    return results


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def maximum_spanning_tree(n: int, matches: list[PairMatch]) -> list[PairMatch]:
    dsu = DisjointSet(n)
    tree: list[PairMatch] = []
    for match in sorted(matches, key=lambda m: m.score, reverse=True):
        if dsu.union(match.i, match.j):
            tree.append(match)
            if len(tree) == n - 1:
                break
    if len(tree) != n - 1:
        components: dict[int, list[int]] = {}
        for idx in range(n):
            components.setdefault(dsu.find(idx), []).append(idx + 1)
        comp_text = "; ".join(",".join(map(str, v)) for v in components.values())
        raise RuntimeError(
            "Not all images could be connected into a single network. "
            f"Current connected groups: {comp_text}. Increase the overlap between adjacent images or check the input images."
        )
    return tree


def choose_reference(n: int, tree: list[PairMatch]) -> int:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for edge in tree:
        cost = 1.0 / max(edge.score, 1.0e-6)
        adjacency[edge.i].append((edge.j, cost))
        adjacency[edge.j].append((edge.i, cost))

    best_node = 0
    best_eccentricity = float("inf")
    for start in range(n):
        distances = [float("inf")] * n
        distances[start] = 0.0
        stack = [(start, -1)]
        while stack:
            node, parent = stack.pop()
            for nxt, cost in adjacency[node]:
                if nxt == parent:
                    continue
                distances[nxt] = distances[node] + cost
                stack.append((nxt, node))
        eccentricity = max(distances)
        if eccentricity < best_eccentricity:
            best_eccentricity = eccentricity
            best_node = start
    return best_node


def compose_global_transforms(
    n: int, tree: list[PairMatch], reference: int
) -> tuple[list[np.ndarray], list[int]]:
    adjacency: list[list[tuple[int, np.ndarray, float]]] = [[] for _ in range(n)]
    for edge in tree:
        H_i_to_j = edge.H_i_to_j
        H_j_to_i = np.linalg.inv(H_i_to_j)
        adjacency[edge.i].append((edge.j, H_i_to_j, edge.score))
        adjacency[edge.j].append((edge.i, H_j_to_i, edge.score))

    # Store H[a][b] as a -> b. If current c is already c -> reference,
    # then neighbor n -> reference = (c -> reference) @ (n -> c).
    transforms: list[np.ndarray | None] = [None] * n
    transforms[reference] = np.eye(3, dtype=np.float64)
    order: list[int] = [reference]
    queue = [reference]

    while queue:
        current = queue.pop(0)
        for neighbor, H_current_to_neighbor, _score in sorted(
            adjacency[current], key=lambda x: x[2], reverse=True
        ):
            if transforms[neighbor] is not None:
                continue
            H_neighbor_to_current = np.linalg.inv(H_current_to_neighbor)
            transforms[neighbor] = transforms[current] @ H_neighbor_to_current  # type: ignore[operator]
            transforms[neighbor] = transforms[neighbor] / transforms[neighbor][2, 2]
            order.append(neighbor)
            queue.append(neighbor)

    if any(H is None for H in transforms):
        raise RuntimeError("Failed to compose the global homographies.")
    return [H for H in transforms if H is not None], order


def transformed_corners(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    return cv2.perspectiveTransform(corners, H)[0].astype(np.float64)


def transformed_center(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    center = np.float32([[[w / 2.0, h / 2.0]]])
    return cv2.perspectiveTransform(center, H)[0, 0].astype(np.float64)


def build_canvas(
    images: list[np.ndarray], transforms: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int], np.ndarray]:
    all_corners = np.vstack([transformed_corners(im, H) for im, H in zip(images, transforms)])
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    translation = np.array(
        [[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    if width <= 0 or height <= 0:
        raise RuntimeError("Failed to calculate a valid output canvas.")
    if width * height > 2_000_000_000:
        raise RuntimeError(
            f"The estimated canvas is excessively large: {width} x {height}. "
            "There may be an incorrect feature connection."
        )

    canvas_transforms = [translation @ H for H in transforms]
    warped_images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    centers: list[np.ndarray] = []

    print(f"\n[4/6] Warping images onto the common canvas: {width} x {height}")
    for idx, (image, H) in enumerate(zip(images, canvas_transforms), start=1):
        print(f"  - Warping image {idx:02d}")
        warped = cv2.warpPerspective(
            image,
            H,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        mask = cv2.warpPerspective(
            source_mask,
            H,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_images.append(warped)
        masks.append(mask)
        centers.append(transformed_center(image, H))

    return warped_images, masks, centers, (width, height), translation


def longest_contiguous_run(values: np.ndarray) -> tuple[int, int]:
    if len(values) == 0:
        raise RuntimeError("No overlap region was found.")
    runs: list[tuple[int, int]] = []
    start = previous = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
        else:
            runs.append((start, previous))
            start = previous = value
    runs.append((start, previous))
    return max(runs, key=lambda item: item[1] - item[0])


def normalized_gray_for_cost(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = to_gray_float(image)
    valid = mask > 0
    out = np.zeros(gray.shape, dtype=np.float32)
    if not np.any(valid):
        return out
    values = gray[valid]
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        return out
    out[valid] = np.clip((gray[valid] - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out


def minimum_error_horizontal_seam(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    *,
    edge_weight: float,
    smoothness_penalty: float,
    max_step: int,
    cost_blur_sigma: float,
) -> np.ndarray:
    overlap = (first_mask > 0) & (second_mask > 0)
    ys, xs = np.where(overlap)
    if len(xs) == 0:
        raise RuntimeError("No overlap region was found between the two images.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    first_gray_full = normalized_gray_for_cost(first_image, first_mask)
    second_gray_full = normalized_gray_for_cost(second_image, second_mask)
    first_gray = first_gray_full[y0 : y1 + 1, x0 : x1 + 1]
    second_gray = second_gray_full[y0 : y1 + 1, x0 : x1 + 1]
    local_overlap = overlap[y0 : y1 + 1, x0 : x1 + 1]

    intensity_difference = cv2.GaussianBlur(
        np.abs(first_gray - second_gray), (0, 0), cost_blur_sigma
    )

    fx = cv2.Sobel(first_gray, cv2.CV_32F, 1, 0, ksize=3)
    fy = cv2.Sobel(first_gray, cv2.CV_32F, 0, 1, ksize=3)
    sx = cv2.Sobel(second_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(second_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.maximum(cv2.magnitude(fx, fy), cv2.magnitude(sx, sy))

    robust_scale = float(np.percentile(edge_strength[local_overlap], 95))
    robust_scale = max(robust_scale, 1.0)
    edge_cost = np.clip(edge_strength / robust_scale, 0.0, 3.0) * 30.0

    cost = intensity_difference + edge_weight * edge_cost
    cost[~local_overlap] = 1.0e12

    valid_columns = np.where(np.any(local_overlap, axis=0))[0]
    run_start, run_end = longest_contiguous_run(valid_columns)
    cost = cost[:, run_start : run_end + 1]
    valid = local_overlap[:, run_start : run_end + 1]

    height, width = cost.shape
    accumulated = np.full((height, width), np.inf, dtype=np.float64)
    previous_y = np.full((height, width), -1, dtype=np.int32)
    first_valid_y = np.where(valid[:, 0])[0]
    accumulated[first_valid_y, 0] = cost[first_valid_y, 0]

    for x in range(1, width):
        valid_y = np.where(valid[:, x])[0]
        for y in valid_y:
            low = max(0, int(y) - max_step)
            high = min(height, int(y) + max_step + 1)
            prior = accumulated[low:high, x - 1]
            if np.all(np.isinf(prior)):
                continue
            candidate_y = np.arange(low, high)
            candidate_cost = prior + smoothness_penalty * np.abs(candidate_y - int(y))
            best_idx = int(np.argmin(candidate_cost))
            accumulated[y, x] = cost[y, x] + candidate_cost[best_idx]
            previous_y[y, x] = low + best_idx

    possible_last = np.where(np.isfinite(accumulated[:, -1]))[0]
    if len(possible_last) == 0:
        raise RuntimeError("Failed to calculate the minimum-error seam.")

    y = int(possible_last[np.argmin(accumulated[possible_last, -1])])
    local_seam = np.full(width, np.nan, dtype=np.float32)
    local_seam[-1] = y
    for x in range(width - 1, 0, -1):
        y = int(previous_y[y, x])
        if y < 0:
            raise RuntimeError("Failed to backtrack the seam.")
        local_seam[x - 1] = y

    seam = np.full(first_mask.shape[1], np.nan, dtype=np.float32)
    first_x = x0 + run_start
    last_x = x0 + run_end
    seam[first_x : last_x + 1] = local_seam + y0
    seam[:first_x] = seam[first_x]
    seam[last_x + 1 :] = seam[last_x]
    return seam


def minimum_error_seam(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    layout: str,
    *,
    edge_weight: float,
    smoothness_penalty: float,
    max_step: int,
    cost_blur_sigma: float,
) -> np.ndarray:
    kwargs = dict(
        edge_weight=edge_weight,
        smoothness_penalty=smoothness_penalty,
        max_step=max_step,
        cost_blur_sigma=cost_blur_sigma,
    )
    if layout == "vertical":
        return minimum_error_horizontal_seam(
            first_image, second_image, first_mask, second_mask, **kwargs
        )

    first_t = np.transpose(first_image, (1, 0)) if first_image.ndim == 2 else np.transpose(first_image, (1, 0, 2))
    second_t = np.transpose(second_image, (1, 0)) if second_image.ndim == 2 else np.transpose(second_image, (1, 0, 2))
    return minimum_error_horizontal_seam(
        first_t, second_t, first_mask.T, second_mask.T, **kwargs
    )


def blend_across_seam(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    seam: np.ndarray,
    *,
    layout: str,
    first_on_negative_side: bool,
    feather_half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = first_mask.shape
    if layout == "vertical":
        coordinate = np.arange(height, dtype=np.float32)[:, None]
        seam_grid = seam.astype(np.float32)[None, :]
    else:
        coordinate = np.arange(width, dtype=np.float32)[None, :]
        seam_grid = seam.astype(np.float32)[:, None]

    alpha_positive = np.clip(
        (coordinate - seam_grid + feather_half_width) / max(2.0 * feather_half_width, 1.0e-6),
        0.0,
        1.0,
    )
    if first_on_negative_side:
        weight_first = 1.0 - alpha_positive
        weight_second = alpha_positive
    else:
        weight_first = alpha_positive
        weight_second = 1.0 - alpha_positive

    first_valid = first_mask > 0
    second_valid = second_mask > 0
    only_first = first_valid & ~second_valid
    only_second = second_valid & ~first_valid
    neither = ~first_valid & ~second_valid

    weight_first[only_first] = 1.0
    weight_second[only_first] = 0.0
    weight_first[only_second] = 0.0
    weight_second[only_second] = 1.0
    weight_first[neither] = 0.0
    weight_second[neither] = 0.0

    total = np.maximum(weight_first + weight_second, 1.0e-6)
    if first_image.ndim == 2:
        output_float = (
            first_image.astype(np.float64) * weight_first
            + second_image.astype(np.float64) * weight_second
        ) / total
    else:
        output_float = (
            first_image.astype(np.float64) * weight_first[..., None]
            + second_image.astype(np.float64) * weight_second[..., None]
        ) / total[..., None]

    if np.issubdtype(first_image.dtype, np.integer):
        info = np.iinfo(first_image.dtype)
        output = np.clip(np.rint(output_float), info.min, info.max).astype(first_image.dtype)
    else:
        output = output_float.astype(first_image.dtype)

    output_mask = ((first_valid | second_valid).astype(np.uint8) * 255)
    return output, output_mask


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, width, height)

    for y in range(height):
        heights = np.where(binary[y] > 0, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(width + 1):
            current = int(heights[x]) if x < width else 0
            start = x
            while stack and stack[-1][1] > current:
                index, bar_height = stack.pop()
                area = bar_height * (x - index)
                if area > best_area:
                    best_area = area
                    best = (index, y - bar_height + 1, x, y + 1)
                start = index
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    return best


def to_preview_bgr(image: np.ndarray) -> np.ndarray:
    gray8 = robust_to_uint8(to_gray_float(image))
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


def draw_seam(preview: np.ndarray, record: SeamRecord) -> None:
    palette = [
        (0, 0, 255),      # red
        (0, 255, 255),    # yellow
        (255, 255, 0),    # cyan
        (255, 0, 255),    # magenta
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 128, 255),    # orange
        (255, 128, 0),
    ]
    color = palette[record.color_index % len(palette)]
    seam = record.seam

    if record.layout == "vertical":
        valid = np.isfinite(seam)
        xs = np.arange(preview.shape[1])[valid]
        ys = np.rint(seam[valid]).astype(np.int32)
    else:
        valid = np.isfinite(seam)
        ys = np.arange(preview.shape[0])[valid]
        xs = np.rint(seam[valid]).astype(np.int32)

    inside = (xs >= 0) & (xs < preview.shape[1]) & (ys >= 0) & (ys < preview.shape[0])
    points = np.column_stack([xs[inside], ys[inside]]).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(preview, [points.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)


def save_tiff(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to save TIFF file: {path}")


def mosaic_blend(
    warped_images: list[np.ndarray],
    masks: list[np.ndarray],
    centers: list[np.ndarray],
    order: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[SeamRecord]]:
    print("\n[5/6] Calculating minimum-error seams and performing sequential blending")
    reference = order[0]
    composite = warped_images[reference].copy()
    composite_mask = masks[reference].copy()
    added = [reference]
    seams: list[SeamRecord] = []

    for step, idx in enumerate(order[1:], start=1):
        overlap = (composite_mask > 0) & (masks[idx] > 0)
        if not np.any(overlap):
            raise RuntimeError(
                f"Image {idx+1} does not overlap with the current composite. "
                "Check the global alignment connections."
            )

        nearest = min(added, key=lambda a: float(np.linalg.norm(centers[idx] - centers[a])))
        delta = centers[idx] - centers[nearest]
        layout = "horizontal" if abs(float(delta[0])) >= abs(float(delta[1])) else "vertical"
        first_on_negative = bool(delta[0] > 0) if layout == "horizontal" else bool(delta[1] > 0)

        print(
            f"  - [{step}/{len(order)-1}] Adding image {idx+1:02d} / "
            f"reference image {nearest+1:02d} / {layout} seam"
        )

        seam = minimum_error_seam(
            composite,
            warped_images[idx],
            composite_mask,
            masks[idx],
            layout,
            edge_weight=args.edge_weight,
            smoothness_penalty=args.smoothness_penalty,
            max_step=args.max_seam_step,
            cost_blur_sigma=args.cost_blur_sigma,
        )
        composite, composite_mask = blend_across_seam(
            composite,
            warped_images[idx],
            composite_mask,
            masks[idx],
            seam,
            layout=layout,
            first_on_negative_side=first_on_negative,
            feather_half_width=args.feather_half_width,
        )
        seams.append(SeamRecord(seam=seam, layout=layout, color_index=step - 1))
        added.append(idx)

    return composite, composite_mask, seams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SIFT feature matching, RANSAC-based homography estimation, and "
            "minimum-error seam blending for SEM/microscopy mosaics."
        )
    )
    parser.add_argument("--input-dir", default="input_images")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--ratio-test", type=float, default=0.78)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)
    parser.add_argument("--max-features", type=int, default=15000)
    parser.add_argument("--min-good-matches", type=int, default=10)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--min-scale", type=float, default=0.75)
    parser.add_argument("--max-scale", type=float, default=1.33)
    parser.add_argument("--max-rotation", type=float, default=20.0)
    parser.add_argument("--edge-weight", type=float, default=1.20)
    parser.add_argument("--smoothness-penalty", type=float, default=2.50)
    parser.add_argument("--max-seam-step", type=int, default=5)
    parser.add_argument("--cost-blur-sigma", type=float, default=1.20)
    parser.add_argument("--feather-half-width", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the output folder limited to the three requested TIFF products.
    for old_output in output_dir.glob("SEM_stitched_*"):
        if old_output.is_file():
            old_output.unlink()

    paths = discover_images(input_dir)
    print("===============================================")
    print("SEM automatic mosaic stitching")
    print("SIFT + RANSAC Homography + Minimum-error seam")
    print("===============================================")
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Detected images: {len(paths)}")
    for i, path in enumerate(paths, start=1):
        print(f"  {i:02d}. {path.name}")

    images = [read_image(path) for path in paths]
    shape0 = images[0].shape
    dtype0 = images[0].dtype
    for path, image in zip(paths, images):
        if image.dtype != dtype0:
            raise RuntimeError(
                f"All input images must have the same bit depth/data type. "
                f"Reference={dtype0}, {path.name}={image.dtype}"
            )
        if (image.ndim == 2) != (images[0].ndim == 2):
            raise RuntimeError("Grayscale and color images cannot be mixed in the same run.")
        if image.ndim == 3 and image.shape[2] != images[0].shape[2]:
            raise RuntimeError("Color images with different numbers of channels cannot be mixed.")

    features = compute_features(images, args.max_features)
    matches = pairwise_matches(images, features, args)
    print("\n[3/6] Constructing a maximum spanning tree from the RANSAC connection network")
    tree = maximum_spanning_tree(len(images), matches)
    reference = choose_reference(len(images), tree)
    transforms, order = compose_global_transforms(len(images), tree, reference)
    print(f"  - Automatically selected reference image: {reference+1:02d} ({paths[reference].name})")
    print(f"  - Stitching order: {', '.join(str(i+1) for i in order)}")

    warped_images, masks, centers, canvas_size, _translation = build_canvas(images, transforms)
    final_image, final_mask, seams = mosaic_blend(warped_images, masks, centers, order, args)

    print("\n[6/6] Saving TIFF results")
    full_path = output_dir / "SEM_stitched_full.tif"
    cropped_path = output_dir / "SEM_stitched_cropped.tif"
    preview_path = output_dir / "SEM_stitched_seams_preview.tif"

    save_tiff(full_path, final_image)
    crop = largest_valid_rectangle(final_mask)
    x0, y0, x1, y1 = crop
    cropped = final_image[y0:y1, x0:x1]
    save_tiff(cropped_path, cropped)

    preview = to_preview_bgr(final_image)
    for record in seams:
        draw_seam(preview, record)
    save_tiff(preview_path, preview)

    print("\n===============================================")
    print("Stitching completed")
    print("===============================================")
    print(f"1. Seam preview : {preview_path}")
    print(f"2. Before cropping : {full_path}")
    print(f"3. After cropping : {cropped_path}")
    print("\nNote: SEM_stitched_cropped.tif is recommended for publication.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR]", exc)
        print("\nDetails:")
        traceback.print_exc()
        sys.exit(1)
