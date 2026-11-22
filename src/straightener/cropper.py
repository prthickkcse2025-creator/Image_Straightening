import cv2
import numpy as np


class SmartCropper:
    """
    Safely removes black borders after perspective warping.

    Instead of finding the largest mathematical rectangle,
    this finds the valid image area and trims only border
    regions that are mostly black.
    """

    def __init__(
        self,
        black_threshold=10,
        safety_margin=3
    ):
        self.black_threshold = black_threshold
        self.safety_margin = safety_margin


    def crop(self, image):

        if image is None:
            raise ValueError(
                "Input image is None"
            )

        height, width = image.shape[:2]

        print("\n===== SMART BORDER CROP =====")

        # ==========================================
        # CREATE VALID PIXEL MASK
        # ==========================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Valid pixels = pixels that are not black
        mask = (
            gray > self.black_threshold
        ).astype(np.uint8)

        # ==========================================
        # REMOVE ONLY MOSTLY BLACK OUTER BORDERS
        # ==========================================

        top = 0
        bottom = height
        left = 0
        right = width

        # A row/column must contain enough valid pixels
        # to be considered part of the real image.
        valid_ratio = 0.995

        # ------------------------------------------
        # FIND TOP
        # ------------------------------------------

        while top < height - 1:

            ratio = np.mean(
                mask[top, :]
            )

            if ratio >= valid_ratio:
                break

            top += 1

        # ------------------------------------------
        # FIND BOTTOM
        # ------------------------------------------

        while bottom > top + 1:

            ratio = np.mean(
                mask[bottom - 1, :]
            )

            if ratio >= valid_ratio:
                break

            bottom -= 1

        # ------------------------------------------
        # FIND LEFT
        # ------------------------------------------

        while left < width - 1:

            ratio = np.mean(
                mask[:, left]
            )

            if ratio >= valid_ratio:
                break

            left += 1

        # ------------------------------------------
        # FIND RIGHT
        # ------------------------------------------

        while right > left + 1:

            ratio = np.mean(
                mask[:, right - 1]
            )

            if ratio >= valid_ratio:
                break

            right -= 1

        # ==========================================
        # APPLY SMALL SAFETY MARGIN
        # ==========================================

        margin = self.safety_margin

        top = min(
            top + margin,
            height - 1
        )

        left = min(
            left + margin,
            width - 1
        )

        bottom = max(
            bottom - margin,
            top + 1
        )

        right = max(
            right - margin,
            left + 1
        )

        # ==========================================
        # VALIDATE CROP
        # ==========================================

        crop_width = right - left
        crop_height = bottom - top

        if crop_width < width * 0.5:

            raise RuntimeError(
                "Crop width became too small. "
                "Border detection failed."
            )

        if crop_height < height * 0.5:

            raise RuntimeError(
                "Crop height became too small. "
                "Border detection failed."
            )

        # ==========================================
        # CROP IMAGE
        # ==========================================

        cropped = image[
            top:bottom,
            left:right
        ]

        print(
            f"Crop region: "
            f"x={left}:{right}, "
            f"y={top}:{bottom}"
        )

        print(
            f"Original size: "
            f"{width} x {height}"
        )

        print(
            f"Cropped size: "
            f"{cropped.shape[1]} x "
            f"{cropped.shape[0]}"
        )

        return cropped