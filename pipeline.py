from pathlib import Path
import cv2
import numpy as np

def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur_image(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (5, 5), 0)

def detect_edges(blurred: np.ndarray) -> np.ndarray:
    return cv2.Canny(blurred, 50, 150)

def threshold_image(gray: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return threshold

def find_objects(edges: np.ndarray, original: np.ndarray):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = original.copy()
    objects = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        perimeter = float(cv2.arcLength(contour, True))
        objects.append({
            "area": round(area, 2),
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "perimeter": round(perimeter, 2),
        })
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 0), 3)
        cv2.putText(
            annotated,
            f"A={int(area)}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    objects.sort(key=lambda item: item["area"], reverse=True)
    return annotated, objects

def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")

def run_pipeline(input_path: Path, output_dir: Path):
    output_dir = Path(output_dir)
    image = load_image(input_path)
    gray = to_grayscale(image)
    blurred = blur_image(gray)
    edges = detect_edges(blurred)
    threshold = threshold_image(gray)
    annotated, objects = find_objects(edges, image)

    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_grayscale.png", gray)
    save_image(output_dir / "03_blurred.png", blurred)
    save_image(output_dir / "04_edges.png", edges)
    save_image(output_dir / "05_threshold.png", threshold)
    save_image(output_dir / "06_contours.png", annotated)

    comparison = cv2.hconcat([
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        edges,
        threshold,
        cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY),
    ])
    save_image(output_dir / "07_comparison.png", comparison)

    stats = {
        "Image height": image.shape[0],
        "Image width": image.shape[1],
        "Channels": image.shape[2],
        "Mean grayscale": round(float(gray.mean()), 2),
        "Edge pixels": int(np.count_nonzero(edges)),
        "Threshold foreground pixels": int(np.count_nonzero(threshold == 255)),
    }

    return {"stats": stats, "objects": objects, "object_count": len(objects)}
