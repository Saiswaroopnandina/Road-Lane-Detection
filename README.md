# Plant-Disease-Detection
    import cv2
    import numpy as np

    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(img, lines):
        if lines is None:
            return
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = edges.shape
        roi_vertices = [
            (0, height),
            (width / 2, height / 2),
            (width, height)
        ]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

        line_image = np.zeros_like(frame)
        draw_lines(line_image, lines)
        return cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    def main():
        cap = cv2.VideoCapture(0)  # Use 0 for webcam or 'video.mp4' for a video file
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = process_frame(frame)
            cv2.imshow("Lane Detection", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()
