import cv2
import numpy as np
import pytesseract

def process_image(self, image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle uneven lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove horizontal lines (notebook lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0,0,0), 2)
    
    # Dilate to connect components of rectangles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from top to bottom
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter out small contours
            roi = gray[y:y+h, x:x+w]
            
            # Perform OCR on the ROI
            text = pytesseract.image_to_string(roi).strip()
            self.parse_layer_info(i, text)
            
            # Optional: Draw rectangles on the original image for visualization
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Optional: Save the image with detected rectangles
    cv2.imwrite('detected_layers.png', image)