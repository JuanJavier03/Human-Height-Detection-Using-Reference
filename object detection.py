from ultralytics import YOLO
import cv2

# 1️⃣ Add this global flag
stop_requested = False

# 2️⃣ Add this function for the stop button
def add_stop_button(window_name):
    def mouse_callback(event, x, y, flags, param):
        global stop_requested
        if event == cv2.EVENT_LBUTTONDOWN:
            if 10 <= x <= 110 and 10 <= y <= 50:  # STOP button region
                stop_requested = True
    cv2.setMouseCallback(window_name, mouse_callback)

def main():
    global stop_requested

    # Load model
    model = YOLO('yolov8n.pt')

    # Start video capture
    cap = cv2.VideoCapture(0)

    # 3️⃣ Create named window and add STOP button
    cv2.namedWindow("YOLOv8 Inference")
    add_stop_button("YOLOv8 Inference")

    while cap.isOpened():
        success, frame = cap.read()

        if not success or stop_requested:
            break

        # Run YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # 4️⃣ Draw STOP button on frame
        cv2.rectangle(annotated_frame, (10, 10), (110, 50), (0, 0, 255), -1)
        cv2.putText(annotated_frame, 'STOP', (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show output
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Optional: Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
