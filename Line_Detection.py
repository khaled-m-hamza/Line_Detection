import cv2
import numpy as np


def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(gray, 50, 150)
    return canny


def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[
        (200, height),
        (800, 350),
        (1200, height), ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def houghLines(cropped):
    return cv2.HoughLinesP(cropped, 2, np.pi / 180, 50, minLineLength=40, maxLineGap=250)


def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def display_lines(img, lines):
    image_of_line = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image_of_line, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return image_of_line



def make_points(image, line):
    slo, inter = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.0 / 5)
    x1 = int((y1 - inter) / slo)
    x2 = int((y2 - inter) / slo)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_average = np.average(right_fit, axis=0) if len(right_fit) > 0 else None
    averaged_lines = []
    if left_fit_average is not None:
        averaged_lines.append(make_points(image, left_fit_average))
    if right_fit_average is not None:
        averaged_lines.append(make_points(image, right_fit_average))
    return averaged_lines



cap = cv2.VideoCapture("new.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture("new.mp4")
        continue
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)


    lines = houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    image_of_line = display_lines(frame, averaged_lines)
    
    collect_image = addWeighted(frame,image_of_line)
    cv2.imshow("result", collect_image)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()