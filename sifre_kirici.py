import random
import os
import sys
import time
import urllib.request

import cv2

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import mediapipe as mp


WINDOW_NAME = "Sifre Kirici"
GAME_DURATION = 20
CONFIRM_SECONDS = 0.45
CAMERA_INDEX = 0
HAND_LANDMARKER_MODEL = "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


class HandCounter:
    def __init__(self):
        self.mode = "tasks"
        self.timestamp_ms = 0
        self.landmarker = None
        self.hands = None
        self.solution_hands_module = getattr(getattr(mp, "solutions", None), "hands", None)

        try:
            self.landmarker = self._create_task_landmarker()
        except Exception as exc:
            if self.solution_hands_module is None:
                raise RuntimeError(
                    "MediaPipe Hand Landmarker baslatilamadi ve bu kurulumda "
                    "mp.solutions.hands bulunmuyor. Internet varsa modeli indirmek "
                    "icin programi tekrar calistirin veya hand_landmarker.task dosyasini "
                    "script klasorune koyun."
                ) from exc

            print(f"Hand Landmarker acilamadi, MediaPipe Hands kullaniliyor: {exc}")
            self.mode = "solutions"
            self.hands = self.solution_hands_module.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
            )

    def _create_task_landmarker(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), HAND_LANDMARKER_MODEL)
        if not os.path.exists(model_path):
            print("Hand Landmarker modeli indiriliyor...")
            temp_path = model_path + ".download"
            try:
                urllib.request.urlretrieve(HAND_LANDMARKER_URL, temp_path)
                os.replace(temp_path, model_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        with open(model_path, "rb") as model_file:
            model_buffer = model_file.read()

        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
        if self.hands is not None:
            self.hands.close()

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mode == "tasks":
            self.timestamp_ms += 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
            if not result.hand_landmarks:
                return 0, []

            hands = []
            total_count = 0
            for index, landmarks in enumerate(result.hand_landmarks):
                handedness = "Right"
                if index < len(result.handedness) and result.handedness[index]:
                    handedness = result.handedness[index][0].category_name

                count = self._count_fingers(landmarks, handedness)
                total_count += count
                hands.append({"landmarks": landmarks, "handedness": handedness, "count": count})

            return min(total_count, 10), hands

        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return 0, []

        hands = []
        total_count = 0
        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness = "Right"
            if result.multi_handedness and index < len(result.multi_handedness):
                handedness = result.multi_handedness[index].classification[0].label

            landmarks = hand_landmarks.landmark
            count = self._count_fingers(landmarks, handedness)
            total_count += count
            hands.append({"landmarks": landmarks, "handedness": handedness, "count": count})

        return min(total_count, 10), hands

    @staticmethod
    def _count_fingers(lm, handedness):
        fingers = 0

        # Ayna modunda ve avuc ici kameraya bakarken:
        # Sag elde bas parmak eklemlerin saginda, sol elde eklemlerin solunda kalir.
        if handedness == "Right":
            thumb_is_open = lm[4].x > lm[3].x and lm[4].x > lm[2].x
        else:
            thumb_is_open = lm[4].x < lm[3].x and lm[4].x < lm[2].x

        if thumb_is_open:
            fingers += 1

        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if lm[tip].y < lm[pip].y:
                fingers += 1

        return fingers

    def draw_landmarks(self, frame, hands):
        if not hands:
            return
        h, w = frame.shape[:2]
        for hand in hands:
            landmarks = hand["landmarks"]
            for start, end in HAND_CONNECTIONS:
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)

            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 7, (0, 60, 0), 1)

            wrist = landmarks[0]
            label_x, label_y = int(wrist.x * w) + 12, int(wrist.y * h) - 12
            hand_label = f"{hand['handedness']}: {hand['count']}"
            draw_text(frame, hand_label, label_x, label_y, 0.6, (0, 255, 255), 2)


def draw_text(frame, text, x, y, scale=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_center_text(frame, text, y, scale, color, thickness=3):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (w - tw) // 2)
    draw_text(frame, text, x, y + th // 2, scale, color, thickness)


def draw_button(frame, x, y, w, h, label, key_text, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
    draw_text(frame, key_text, x + 22, y + 46, 1.2, (255, 255, 255), 3)
    draw_text(frame, label, x + 22, y + 92, 0.9, (255, 255, 255), 2)


def draw_menu(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    draw_center_text(frame, "SIFRE KIRICI", 105, 1.8, (0, 255, 255), 4)
    draw_center_text(frame, "Mod secmek icin klavyeden 1 veya 2'ye basin", 158, 0.75, (255, 255, 255), 2)

    button_w = min(330, w // 2 - 70)
    button_h = 130
    gap = 35
    total_w = button_w * 2 + gap
    start_x = (w - total_w) // 2
    y = max(210, h // 2 - 50)
    draw_button(frame, start_x, y, button_w, button_h, "KOLAY (5 Rakam)", "1", (20, 130, 65))
    draw_button(frame, start_x + button_w + gap, y, button_w, button_h, "ZOR (10 Rakam)", "2", (35, 65, 170))

    draw_center_text(frame, "Cikis: Q", h - 70, 0.75, (220, 220, 220), 2)


def make_code(length, max_digit):
    code = []
    for _ in range(length):
        digit = random.randint(1, max_digit)
        while code and digit == code[-1]:
            digit = random.randint(1, max_digit)
        code.append(digit)
    return code


def draw_game_hud(frame, code, index, remaining, finger_count, level_name):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 128), (25, 25, 25), -1)
    cv2.rectangle(frame, (0, 128), (w, 132), (0, 255, 255), -1)

    visible_code = []
    for i, digit in enumerate(code):
        if i < index:
            visible_code.append("*")
        elif i == index:
            visible_code.append(f"[{digit}]")
        else:
            visible_code.append(str(digit))
    code_text = "SIFRE: " + "-".join(visible_code)

    draw_text(frame, f"LEVEL: {level_name}", 24, 36, 0.75, (255, 255, 255), 2)
    draw_text(frame, code_text, 24, 84, 0.9, (0, 255, 255), 2)
    draw_text(frame, f"SURE: {max(0, remaining):04.1f}", w - 210, 42, 0.8, (0, 255, 0), 2)

    if index < len(code):
        draw_text(frame, f"SIRADAKI: {code[index]}", w - 220, 92, 0.75, (255, 255, 255), 2)

    count_text = f"TOPLAM PARMAK: {finger_count}"
    cv2.rectangle(frame, (24, h - 126), (390, h - 24), (0, 120, 75), -1)
    cv2.rectangle(frame, (24, h - 126), (390, h - 24), (255, 255, 255), 3)
    draw_text(frame, count_text, 45, h - 62, 1.05, (255, 255, 255), 3)

    big_text = str(finger_count)
    (tw, th), _ = cv2.getTextSize(big_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 7)
    cv2.rectangle(frame, (w - 170, h - 180), (w - 28, h - 28), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 170, h - 180), (w - 28, h - 28), (0, 255, 255), 3)
    draw_text(frame, big_text, w - 99 - tw // 2, h - 92 + th // 2, 3.0, (0, 255, 255), 7)


def draw_result(frame, won, level_name):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if won:
        draw_center_text(frame, f"KAZANDINIZ - {level_name}", h // 2 - 40, 1.45, (0, 255, 0), 4)
    else:
        draw_center_text(frame, "KAYBETTINIZ", h // 2 - 40, 1.7, (0, 0, 255), 4)

    draw_center_text(frame, "Menu: M    Cikis: Q", h // 2 + 45, 0.8, (255, 255, 255), 2)


def reset_game(length, level_name, max_digit):
    return {
        "state": "playing",
        "level_name": level_name,
        "code": make_code(length, max_digit),
        "index": 0,
        "start_time": time.time(),
        "matched_since": None,
        "last_required": None,
        "won": False,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Calisan dosya: {os.path.join(script_dir, os.path.basename(__file__))}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Kamera acilamadi. CAMERA_INDEX degerini veya kamera izinlerini kontrol edin.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    counter = HandCounter()
    game = {"state": "menu"}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            finger_count, hands = counter.detect(frame)
            counter.draw_landmarks(frame, hands)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if game["state"] == "menu":
                draw_menu(frame)
                if key == ord("1"):
                    game = reset_game(5, "KOLAY", 5)
                elif key == ord("2"):
                    game = reset_game(10, "ZOR", 10)

            elif game["state"] == "playing":
                elapsed = time.time() - game["start_time"]
                remaining = GAME_DURATION - elapsed

                if remaining <= 0:
                    game["state"] = "result"
                    game["won"] = False
                else:
                    required = game["code"][game["index"]]

                    if finger_count == required:
                        if game["matched_since"] is None or game["last_required"] != required:
                            game["matched_since"] = time.time()
                            game["last_required"] = required
                        elif time.time() - game["matched_since"] >= CONFIRM_SECONDS:
                            game["index"] += 1
                            game["matched_since"] = None
                            game["last_required"] = None

                            if game["index"] >= len(game["code"]):
                                game["state"] = "result"
                                game["won"] = True
                    else:
                        game["matched_since"] = None
                        game["last_required"] = None

                draw_game_hud(
                    frame,
                    game["code"],
                    game["index"],
                    GAME_DURATION - (time.time() - game["start_time"]),
                    finger_count,
                    game["level_name"],
                )

            elif game["state"] == "result":
                draw_result(frame, game["won"], game["level_name"])
                if key == ord("m"):
                    game = {"state": "menu"}

            cv2.imshow(WINDOW_NAME, frame)

    finally:
        counter.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Hata: {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Dosya bulunamadi: {exc}")
        print("PowerShell'de once scriptin oldugu klasore gecin:")
        print(r'cd "C:\Users\eyups\Desktop\şenlik oyunları"')
        print(r"python .\sifre_kirici.py")
        sys.exit(1)
