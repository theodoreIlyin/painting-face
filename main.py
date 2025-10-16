import cv2

def main(camera_index=0):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Не вдалося завантажити каскад облич.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Не вдалося відкрити камеру.")

    scale = 0.75

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        inv = 1.0 / scale
        for (x, y, w, h) in faces:
            x1 = int(x * inv); y1 = int(y * inv)
            x2 = int((x + w) * inv); y2 = int((y + h) * inv)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Faces: {len(faces)}  |  q - quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Face Detection (Haar cascade)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
