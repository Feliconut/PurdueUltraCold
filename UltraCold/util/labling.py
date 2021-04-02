import cv2
import numpy as np


def find_rect(raw_img):
    """
    @brief Finds rectangular bounds for intense area of an image.
    The high intensity area should have a clear boundary.
    
    @param raw_img the OD img input.
    For higher clarity it's recommended to use a OD mean img.

    @return A list of rectangles, where each rectangle = [xpos, ypos, width, height]
    """
    img = raw_img.copy()
    img = np.clip(img, 0, 100)
    img = cv2.GaussianBlur(img, (5, 5), 2)
    m, M = img.min(), img.max()

    img = ((img - m) / (M - m) * 255).astype(np.uint8)
    img = cv2.threshold(img, 50, 200, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

    return [cv2.boundingRect(c) for c in contours if 9e4/2 > cv2.contourArea(c) > 3000]


def approx_trap_region(OD_img, padding=5):
    """
    @brief Finds a trap region slicer for an OD image.

    @param OD_img An OD image to find trap region for.
    The image should have a clear, visually distinguishable trap region.

    @param padding Extension to the trap region, in pixels.

    @return A slicer for OD image, representing the trap region.
    """
    rects = find_rect(OD_img)
    assert len(rects) == 1
    x, y, w, h = rects[0]
    b = padding
    return (
        slice(y - b, y + h + b),
        slice(x - b, x + w + b),
    )
