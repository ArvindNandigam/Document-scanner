import cv2
import numpy as np
import pytesseract

def order_points(pts):
    # Order points in TL, TR, BR, BL order
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]    # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]    # bottom-right has largest sum

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)] # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)] # bottom-left has largest diff
    return rect

def scan_image_to_text(image):
    """
    Detects a document in the image, warps perspective, enhances it, 
    and runs OCR to return extracted text.
    """
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edged = cv2.Canny(gray, 75, 200)
    # Find contours
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # If our approximated contour has four points, we assume it's the document
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is not None:
        # Warp the image to get a top-down view of the document
        pts = doc_cnt.reshape(4, 2)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        # Compute width and height of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    else:
        # If no contour is detected, just use the original image
        warped = orig

    # Convert to grayscale (again) and apply threshold to make text more clear
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Use Otsu's thresholding
    _, thresh = cv2.threshold(warped_gray, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Optionally, reduce noise
    thresh = cv2.medianBlur(thresh, 3)

    # OCR: extract text from the processed (thresholded) image
    text = pytesseract.image_to_string(thresh)
    return text
