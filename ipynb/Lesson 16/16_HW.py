# %% Cell
# Команда обеспечивает вывод графиков в Google Colaboratory
# %matplotlib inline

# import gdown
# gdown.download('https://storage.yandexcloud.net/academy.ai/friends.jpg', None, quiet=True)


# %% Cell
import math
import sys
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from cv2.typing import Rect
from loguru import logger
from matplotlib import pyplot


# %% Cell
class CVObject:
    def __init__(self, image: np.ndarray, rect: Rect):
        self.image = image
        self.x, self.y, self.w, self.h = rect
        self.center = (int(self.x + 0.5 * self.w), int(self.y + 0.5 * self.h))


class Eye(CVObject):
    def __init__(self, image: np.ndarray, rect: Rect):
        super().__init__(image, rect)
        self.radius = int(0.3 * (self.w + self.h))


class Face(CVObject):
    def __init__(self, image: np.ndarray, rect: Rect):
        super().__init__(image, rect)
        self.axes = (self.w // 2, self.h // 2)
        self.face_zone = self.image[self.y : self.y + self.h, self.x : self.x + self.w]
        self.eyes = self.detect_eyes()
        self.eyes_zone = (
            self.face_zone[
                self.eyes[0].y : self.eyes[1].y, self.eyes[0].x : self.eyes[1].x
            ]
            if len(self.eyes) == 2
            else None
        )

    def draw_face_rect(self):
        cv2.rectangle(
            self.image,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            (0, 0, 255),
            2,
        )

    def draw_zone_eyes(self):
        cv2.rectangle(
            self.image,
            (self.eyes[0].x + self.x, self.eyes[0].y + self.y),
            (
                self.eyes[1].x + self.x + self.eyes[1].w,
                self.eyes[1].y + self.y + self.eyes[1].h,
            ),
            (255, 0, 0),
            2,
        )

    def draw_face_circle(self):
        cv2.ellipse(self.image, self.center, self.axes, 0, 0, 360, (0, 255, 0), 2)

    def draw_eyes(self):
        for eye in self.eyes:
            cv2.circle(
                self.image,
                (eye.center[0] + self.x, eye.center[1] + self.y),
                eye.radius,
                (0, 255, 0),
                2,
            )

    def rotate_glasses(
        self,
        glasses_img: np.ndarray,
        glasses_width: int,
        glasses_height: int,
        dx: int,
        dy: int,
        diagonal: int,
    ):
        # Calculate angle between eyes
        angle = math.atan2(dy, dx) * 180 / math.pi * -1
        logger.debug(f"Angle between eyes: {angle}")

        # Rotate the glasses
        # Increase the size of the rotation canvas to prevent cutting off
        rotation_matrix = cv2.getRotationMatrix2D(
            (diagonal // 2, diagonal // 2), angle, 1
        )
        rotated_glasses = cv2.warpAffine(
            glasses_img, rotation_matrix, (diagonal, diagonal)
        )
        return rotated_glasses

    def place_glasses(self, glasses_img: np.ndarray, rotate=False):
        if len(self.eyes) != 2:
            return

        # Calculate the width and height of the glasses
        dx = self.eyes[1].center[0] - self.eyes[0].center[0]
        dy = self.eyes[1].center[1] - self.eyes[0].center[1]

        if rotate:
            eye_distance = math.sqrt(dx**2 + dy**2)
        else:
            eye_distance = int(abs(self.eyes[1].center[0] - self.eyes[0].center[0]))

        glasses_width = int(eye_distance * 2.5)
        glasses_height = int(
            glasses_width * glasses_img.shape[0] / glasses_img.shape[1]
        )

        # Resize the glasses image
        _glasses = cv2.resize(
            glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA
        )

        # Rotate the glasses
        if rotate:
            diagonal = int(math.sqrt(glasses_width**2 + glasses_height**2))
            _glasses = self.rotate_glasses(
                _glasses, glasses_width, glasses_height, dx, dy, diagonal
            )

        # Calculate the position to place the glasses
        center_x = (self.eyes[0].center[0] + self.eyes[1].center[0]) // 2
        center_y = (self.eyes[0].center[1] + self.eyes[1].center[1]) // 2
        glasses_x = center_x - glasses_width // 2
        glasses_y = center_y - glasses_height // 2

        # Ensure the glasses fit within the face zone
        glasses_x = max(0, min(glasses_x, self.w - glasses_width))
        glasses_y = max(0, min(glasses_y, self.h - glasses_height))

        # Create a region of interest (ROI) in the face zone
        roi = self.face_zone[
            glasses_y : glasses_y + glasses_height,
            glasses_x : glasses_x + glasses_width,
        ]

        # Ensure the glasses and ROI have the same dimensions
        roi_height, roi_width = roi.shape[:2]
        _glasses = _glasses[:roi_height, :roi_width]

        # Split the image into color channels and alpha channel
        glasses_bgr = _glasses[:, :, :3]
        glasses_alpha = _glasses[:, :, 3]

        # Create a binary mask from the alpha channel
        _, mask = cv2.threshold(glasses_alpha, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Convert masks to 3-channel
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_inv_3channel = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        # Apply the mask
        glasses_fg = cv2.bitwise_and(glasses_bgr, mask_3channel)
        roi_bg = cv2.bitwise_and(roi, mask_inv_3channel)

        # Combine the glasses with the ROI
        dst = cv2.add(roi_bg, glasses_fg)

        # Place the result back in the face zone
        self.face_zone[
            glasses_y : glasses_y + roi_height, glasses_x : glasses_x + roi_width
        ] = dst

    def blur_face_except_eyes(self, blur_amount: int = 45):
        blurred_face = self.face_zone.copy()
        blurred_face = cv2.GaussianBlur(blurred_face, (blur_amount, blur_amount), 0)

        # unblurred
        mask = np.zeros(self.face_zone.shape[:2], dtype=np.uint8)
        for eye in self.eyes:
            cv2.circle(mask, eye.center, eye.radius, 255, -1)
            cv2.copyTo(self.face_zone, mask, blurred_face)

        self.image[self.y : self.y + self.h, self.x : self.x + self.w] = blurred_face

    def detect_eyes(self) -> list[Eye]:
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        eyes = eye_cascade.detectMultiScale(self.face_zone)

        if len(eyes) > 2:
            # Delete 3 eyes
            eyes = sorted(eyes, key=lambda e: e[1])[:2]
            # Sort by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])

        return [Eye(self.face_zone, eye) for eye in eyes]

    @classmethod
    def detect_faces(cls, image: np.ndarray) -> list[Self]:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [cls(image, face) for face in faces]


def search_img(file_name: str, asset_paths: list[Path]) -> Path:
    for asset_path in asset_paths:
        img_path = asset_path / file_name
        if img_path.exists():
            return img_path
    raise FileNotFoundError(f"Image file not found: {file_name}")


def show_result(img_source, img_result):
    # Финальная визуализация
    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(15, 8))
    ax1.imshow(cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
    ax1.set_title("Исходное изображение")

    ax2.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    ax2.set_title("Распознанные лица")

    pyplot.show()


def main(debug=False):
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    asset_paths = [Path("assets"), Path("../assets")]
    celeb_img_path = search_img("celeb.jpg", asset_paths)
    celeb_img_path = search_img("celeb2.webp", asset_paths)
    glass_img_path = search_img("glass.png", asset_paths)

    if celeb_img_path is None:
        raise FileNotFoundError(f"Image file not found! {celeb_img_path}")
    if glass_img_path is None:
        raise FileNotFoundError(f"Image file not found!{glass_img_path}")

    # Загрузка изображения
    celeb_img = cv2.imread(str(celeb_img_path))
    celeb_img_result = celeb_img.copy()
    glass_img = cv2.imread(str(glass_img_path), cv2.IMREAD_UNCHANGED)

    # Распознавание лиц
    faces = Face.detect_faces(celeb_img_result)
    logger.info(f"Was found {len(faces)} faces")
    for face in faces:
        face.place_glasses(glasses_img=glass_img)
        face.blur_face_except_eyes()
        if debug:
            face.draw_face_rect()
            face.draw_zone_eyes()
        else:
            face.draw_face_circle()
            face.draw_eyes()

    show_result(celeb_img, celeb_img_result)


if __name__ == "__main__":
    main(debug=True)
