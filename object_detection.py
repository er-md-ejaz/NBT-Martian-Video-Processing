import cv2
from ultralytics import YOLO


model = YOLO("yolov8m.pt")


# cap = cv2.VideoCapture("hotel_video.mp4")
cap = cv2.VideoCapture(0)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)


out = cv2.VideoWriter("output_yolo.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow('YOLO Detection', annotated_frame)

    out.write(annotated_frame)

    for r in results:
        boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"Detected: {label} with confidence {conf:.5f}")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
out.release()
cv2.destroyAllWindows()
