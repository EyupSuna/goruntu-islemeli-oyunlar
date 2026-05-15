import math
import os
import random
import sys
import time
import urllib.request

import cv2

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import mediapipe as mp


WINDOW_NAME = "Kirmizi Isik Yesil Isik"
CAMERA_INDEX = 0
GAME_DURATION = 20.0
RED_MOTION_THRESHOLD = 0.026
RED_GRACE_SECONDS = 0.25
RED_STRIKE_LIMIT = 2
RED_INSTANT_MOTION_THRESHOLD = 0.055
GREEN_MESSAGE_SECONDS = 1.15
GREEN_MOTION_THRESHOLD = 0.004
PROGRESS_GAIN = 7.0
POSE_MODEL = "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)

TRACKED_LANDMARKS = (
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
)


class PoseTracker:
    def __init__(self):
        self.mode = "tasks"
        self.timestamp_ms = 0
        self.landmarker = None
        self.pose = None
        self.solution_pose_module = getattr(getattr(mp, "solutions", None), "pose", None)

        try:
            self.landmarker = self._create_task_landmarker()
        except Exception as exc:
            if self.solution_pose_module is None:
                raise RuntimeError(
                    "MediaPipe PoseLandmarker baslatilamadi ve bu kurulumda "
                    "mp.solutions.pose bulunmuyor. Internet varsa modeli indirmek "
                    "icin programi tekrar calistirin veya pose_landmarker_lite.task "
                    "dosyasini script klasorune koyun."
                ) from exc

            print(f"PoseLandmarker acilamadi, MediaPipe Pose kullaniliyor: {exc}")
            self.mode = "solutions"
            self.pose = self.solution_pose_module.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
            )

    def _create_task_landmarker(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), POSE_MODEL)
        if not os.path.exists(model_path):
            print("Pose Landmarker modeli indiriliyor...")
            temp_path = model_path + ".download"
            try:
                urllib.request.urlretrieve(POSE_MODEL_URL, temp_path)
                os.replace(temp_path, model_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        with open(model_path, "rb") as model_file:
            model_buffer = model_file.read()

        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_buffer)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.65,
            min_pose_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
        if self.pose is not None:
            self.pose.close()

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mode == "tasks":
            self.timestamp_ms += 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
            if not result.pose_landmarks:
                return None
            return result.pose_landmarks[0]

        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks.landmark

    @staticmethod
    def draw(frame, landmarks):
        if landmarks is None:
            return

        h, w = frame.shape[:2]
        for start, end in POSE_CONNECTIONS:
            if start >= len(landmarks) or end >= len(landmarks):
                continue
            if not is_visible(landmarks[start]) or not is_visible(landmarks[end]):
                continue
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 240, 240), 2)

        for lm in landmarks:
            if not is_visible(lm):
                continue
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 6, (0, 70, 0), 1)


def is_visible(landmark):
    return getattr(landmark, "visibility", 1.0) >= 0.45


def pose_motion(previous, current):
    if previous is None or current is None:
        return 0.0

    total = 0.0
    count = 0
    for index in TRACKED_LANDMARKS:
        if index >= len(previous) or index >= len(current):
            continue
        if not is_visible(previous[index]) or not is_visible(current[index]):
            continue

        dx = current[index].x - previous[index].x
        dy = current[index].y - previous[index].y
        total += math.sqrt(dx * dx + dy * dy)
        count += 1

    if count == 0:
        return 0.0
    return total / count


def draw_text(frame, text, x, y, scale=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_center_text(frame, text, y, scale, color, thickness=4):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (w - tw) // 2)
    draw_text(frame, text, x, y + th // 2, scale, color, thickness)


def draw_traffic_light(frame, light):
    x, y = 82, 76
    color = (0, 230, 0) if light == "green" else (0, 0, 255)
    label = "YESIL" if light == "green" else "KIRMIZI"

    cv2.rectangle(frame, (28, 24), (246, 168), (35, 35, 35), -1)
    cv2.rectangle(frame, (28, 24), (246, 168), (255, 255, 255), 3)
    cv2.circle(frame, (x + 24, y), 45, color, -1)
    cv2.circle(frame, (x + 24, y), 47, (255, 255, 255), 3)
    draw_text(frame, label, 132, 88, 0.78, color, 2)
    draw_text(frame, "ISIK", 132, 126, 0.62, (255, 255, 255), 2)


def draw_progress(frame, progress):
    h, w = frame.shape[:2]
    left, top = 42, h - 82
    right, bottom = w - 42, h - 36
    bar_w = right - left
    fill_w = int(bar_w * max(0.0, min(1.0, progress)))

    cv2.rectangle(frame, (left, top), (right, bottom), (35, 35, 35), -1)
    cv2.rectangle(frame, (left, top), (left + fill_w, bottom), (0, 190, 80), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 3)
    draw_text(frame, f"ILERLEME: %{int(progress * 100)}", left + 12, top - 14, 0.75, (255, 255, 255), 2)


def draw_status_chip(frame, text, light):
    h, w = frame.shape[:2]
    color = (0, 255, 0) if light == "green" else (0, 0, 255)
    bg_color = (0, 70, 0) if light == "green" else (0, 0, 80)
    scale = 1.15 if light == "green" else 1.35
    thickness = 4
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad_x, pad_y = 28, 18
    x1 = (w - tw) // 2 - pad_x
    y1 = 138
    x2 = (w + tw) // 2 + pad_x
    y2 = y1 + th + pad_y * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    draw_text(frame, text, (w - tw) // 2, y1 + pad_y + th, scale, color, thickness)


def draw_hud(frame, light, remaining, progress, motion, show_green_message=False):
    h, w = frame.shape[:2]
    draw_traffic_light(frame, light)
    draw_progress(frame, progress)

    timer_color = (0, 255, 0) if remaining > 6 else (0, 180, 255)
    draw_text(frame, f"SURE: {max(0.0, remaining):04.1f}", w - 230, 52, 0.9, timer_color, 3)
    draw_text(frame, f"HAREKET: {motion:.3f}", w - 230, 96, 0.66, (255, 255, 255), 2)

    if light == "red":
        draw_status_chip(frame, "DUR!", light)
    elif show_green_message:
        draw_status_chip(frame, "HAREKET ET!", light)


def draw_result(frame, result):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    if result == "won":
        draw_center_text(frame, "KAZANDINIZ", h // 2 - 44, 1.8, (0, 255, 0), 5)
    elif result == "eliminated":
        draw_center_text(frame, "ELENDINIZ", h // 2 - 44, 1.8, (0, 0, 255), 5)
    else:
        draw_center_text(frame, "SURE BITTI", h // 2 - 44, 1.8, (0, 180, 255), 5)

    draw_center_text(frame, "Tekrar: R    Cikis: Q", h // 2 + 52, 0.8, (255, 255, 255), 2)


def next_switch_time(now, light):
    if light == "green":
        return now + random.uniform(2.0, 4.0)
    return now + random.uniform(1.0, 2.2)


def new_game():
    now = time.time()
    return {
        "state": "playing",
        "start_time": now,
        "last_frame_time": now,
        "light": "green",
        "next_switch": next_switch_time(now, "green"),
        "green_message_until": now + GREEN_MESSAGE_SECONDS,
        "red_started_at": None,
        "red_motion_strikes": 0,
        "progress": 0.0,
        "previous_landmarks": None,
        "result": None,
    }


def main():
    print(f"Calisan dosya: {os.path.abspath(__file__)}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Kamera acilamadi. CAMERA_INDEX degerini veya kamera izinlerini kontrol edin.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = PoseTracker()
    game = new_game()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            now = time.time()
            dt = max(0.001, now - game["last_frame_time"])
            game["last_frame_time"] = now

            landmarks = tracker.detect(frame)
            tracker.draw(frame, landmarks)

            motion = pose_motion(game["previous_landmarks"], landmarks)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                game = new_game()

            if game["state"] == "playing":
                elapsed = now - game["start_time"]
                remaining = GAME_DURATION - elapsed

                if now >= game["next_switch"]:
                    game["light"] = "red" if game["light"] == "green" else "green"
                    game["next_switch"] = next_switch_time(now, game["light"])
                    game["red_motion_strikes"] = 0
                    game["red_started_at"] = now if game["light"] == "red" else None
                    game["green_message_until"] = (
                        now + GREEN_MESSAGE_SECONDS if game["light"] == "green" else 0.0
                    )

                if game["light"] == "red":
                    red_time = 0.0 if game["red_started_at"] is None else now - game["red_started_at"]
                    if red_time >= RED_GRACE_SECONDS and motion > RED_INSTANT_MOTION_THRESHOLD:
                        game["state"] = "result"
                        game["result"] = "eliminated"
                    elif red_time >= RED_GRACE_SECONDS and motion > RED_MOTION_THRESHOLD:
                        game["red_motion_strikes"] += 1
                    else:
                        game["red_motion_strikes"] = max(0, game["red_motion_strikes"] - 1)

                    if game["red_motion_strikes"] >= RED_STRIKE_LIMIT:
                        game["state"] = "result"
                        game["result"] = "eliminated"
                elif game["light"] == "green" and motion > GREEN_MOTION_THRESHOLD:
                    game["red_motion_strikes"] = 0
                    game["progress"] += motion * dt * PROGRESS_GAIN
                    game["progress"] = min(1.0, game["progress"])

                if game["progress"] >= 1.0:
                    game["state"] = "result"
                    game["result"] = "won"
                elif remaining <= 0 and game["state"] == "playing":
                    game["state"] = "result"
                    game["result"] = "timeout"

                draw_hud(
                    frame,
                    game["light"],
                    remaining,
                    game["progress"],
                    motion,
                    now < game["green_message_until"],
                )
            else:
                draw_hud(frame, game["light"], 0.0, game["progress"], motion, False)
                draw_result(frame, game["result"])

            game["previous_landmarks"] = landmarks
            cv2.imshow(WINDOW_NAME, frame)

    finally:
        tracker.close()
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
        print(r"python .\squid_game.py")
        sys.exit(1)
