import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace


def Image_Results():
    input_video_path = "./Dataset/Sec Growth DataScience staff meeting Sep 14 2022 [rOqgRiNMVqg].f398.mp4"
    cap = cv2.VideoCapture(input_video_path)
    start_frame = 300
    end_frame = 400
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_emotions = []
        try:
            predictions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True,
                                           detector_backend='yolov8', silent=True)
            if not isinstance(predictions, list):
                predictions = [predictions]
            for face in predictions:
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                dominant_emotion = face['dominant_emotion']
                confidence = face['emotion'][dominant_emotion]
                label_text = f"{dominant_emotion.capitalize()}: {int(confidence)}%"
                frame_emotions.append(dominant_emotion.capitalize())
                # Draw boundary box and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            pass
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()


def Sample_Images():
    Orig = np.load('Image.npy', allow_pickle=True)
    ind = [100, 115, 145, 155, 200, 500]
    fig, ax = plt.subplots(2, 3)
    plt.suptitle("Sample Images ")
    plt.subplot(2, 3, 1)
    plt.title('Image-1')
    plt.imshow(Orig[ind[0]])
    plt.subplot(2, 3, 2)
    plt.title('Image-2')
    plt.imshow(Orig[ind[1]])
    plt.subplot(2, 3, 3)
    plt.title('Image-3')
    plt.imshow(Orig[ind[2]])
    plt.subplot(2, 3, 4)
    plt.title('Image-4')
    plt.imshow(Orig[ind[3]])
    plt.subplot(2, 3, 5)
    plt.title('Image-5')
    plt.imshow(Orig[ind[4]])
    plt.subplot(2, 3, 6)
    plt.title('Image-6')
    plt.imshow(Orig[ind[5]])
    plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
