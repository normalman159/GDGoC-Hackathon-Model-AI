import cv2
import mediapipe as mp

from model_run import I3DModel

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands = 2)


def main() :
    cap = cv2.VideoCapture(0)
    frames = []

    weight = 'FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    weights_RGB = 'rgb_imagenet.pt'

    i3d = I3DModel(weights_RGB=weights_RGB, weight=weight, num_classes=100)

    while True:
        ret, frame = cap.read()
        if not ret: return
        # frame = cv2.flip(frame, 1)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (800, 600))
        result = hands.process(frame)

        if result.multi_hand_landmarks:

            frames.append(frame)
            print("Detected Hand!!!!!!!")

        else :
            if len(frames) > 0:
                print(len(frames), "CAN DO SOMETHING HERE")

            result = i3d.run(frames)
            print(result)
            frames = []

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()