from pathlib import Path
import cv2
import numpy as np

def create_sample_image(path: Path) -> None:
    path = Path(path)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (245, 245, 245)

    # Rectangle
    cv2.rectangle(img, (60, 70), (250, 220), (30, 130, 220), -1)

    # Circle
    cv2.circle(img, (430, 150), 75, (40, 180, 80), -1)

    # Triangle
    triangle = np.array([[280, 340], [400, 250], [520, 360]], dtype=np.int32)
    cv2.fillPoly(img, [triangle], (210, 80, 70))

    # Smaller rectangle
    cv2.rectangle(img, (90, 300), (220, 410), (90, 90, 200), -1)

    # Decorative lines
    cv2.line(img, (35, 445), (600, 445), (40, 40, 40), 4)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
