import numpy as np
import cv2


class GeometryRectifier:

    def __init__(self, focal_length=None):
        self.focal_length = focal_length


    @staticmethod
    def _normalize(vector):

        vector = np.asarray(
            vector,
            dtype=np.float64
        )

        norm = np.linalg.norm(vector)

        if norm < 1e-10:
            raise ValueError(
                "Cannot normalize zero vector"
            )

        return vector / norm


    def build_intrinsic_matrix(
        self,
        width,
        height
    ):

        focal_length = (
            float(self.focal_length)
            if self.focal_length is not None
            else float(max(width, height))
        )

        cx = width / 2.0
        cy = height / 2.0

        return np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0]
            ],
            dtype=np.float64
        )


    def build_rectification_homography(
        self,
        vertical_vp,
        horizontal_vp_1,
        horizontal_vp_2,
        width,
        height
    ):

        # -----------------------------------------
        # NORMALIZE VANISHING DIRECTIONS
        # -----------------------------------------

        vertical = self._normalize(
            vertical_vp
        )

        h1 = self._normalize(
            horizontal_vp_1
        )

        h2 = self._normalize(
            horizontal_vp_2
        )

        # -----------------------------------------
        # CAMERA INTRINSICS
        # -----------------------------------------

        K = self.build_intrinsic_matrix(
            width,
            height
        )

        f = K[0, 0]

        # -----------------------------------------
        # 1. ROLL CORRECTION
        # -----------------------------------------
        #
        # The vertical VP gives the camera roll.
        #
        # We calculate a small correction that makes
        # the vertical architectural direction upright.
        # -----------------------------------------

        roll = np.arctan2(
            vertical[0],
            vertical[1]
        )

        # Do not allow accidental 180 degree flips.
        if roll > np.pi / 2:
            roll -= np.pi

        if roll < -np.pi / 2:
            roll += np.pi

        # Controlled roll correction.
        roll_correction = -roll

        max_roll = np.deg2rad(15.0)

        roll_correction = np.clip(
            roll_correction,
            -max_roll,
            max_roll
        )

        cos_r = np.cos(
            roll_correction
        )

        sin_r = np.sin(
            roll_correction
        )

        R_roll = np.array(
            [
                [cos_r, -sin_r, 0.0],
                [sin_r, cos_r, 0.0],
                [0.0, 0.0, 1.0]
            ],
            dtype=np.float64
        )

        # -----------------------------------------
        # 2. VERTICAL PERSPECTIVE CORRECTION
        # -----------------------------------------
        #
        # Use the Z component of the vertical VP
        # to estimate camera pitch.
        #
        # Clamp the correction so that the image
        # remains natural.
        # -----------------------------------------

        pitch = np.arctan2(
            vertical[2],
            abs(vertical[1])
        )

        pitch_correction = -pitch

        max_pitch = np.deg2rad(12.0)

        pitch_correction = np.clip(
            pitch_correction,
            -max_pitch,
            max_pitch
        )

        cos_p = np.cos(
            pitch_correction
        )

        sin_p = np.sin(
            pitch_correction
        )

        R_pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_p, -sin_p],
                [0.0, sin_p, cos_p]
            ],
            dtype=np.float64
        )

        # -----------------------------------------
        # COMBINE CORRECTIONS
        # -----------------------------------------

        R = R_pitch @ R_roll

        K_inv = np.linalg.inv(
            K
        )

        H = K @ R @ K_inv

        # Normalize homography.
        H = H / H[2, 2]

        print(
            "\n===== CONTROLLED CORRECTION ====="
        )

        print(
            f"Roll correction: "
            f"{np.rad2deg(roll_correction):.3f} degrees"
        )

        print(
            f"Pitch correction: "
            f"{np.rad2deg(pitch_correction):.3f} degrees"
        )

        return H, R, K


    def warp_image(
        self,
        image,
        H,
        output_size=None
    ):

        height, width = image.shape[:2]

        # -----------------------------------------
        # ORIGINAL CORNERS
        # -----------------------------------------

        corners = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ],
            dtype=np.float32
        ).reshape(
            -1,
            1,
            2
        )

        # Transform corners.

        transformed = cv2.perspectiveTransform(
            corners,
            H
        )

        points = transformed.reshape(
            -1,
            2
        )

        min_x = np.floor(
            points[:, 0].min()
        )

        max_x = np.ceil(
            points[:, 0].max()
        )

        min_y = np.floor(
            points[:, 1].min()
        )

        max_y = np.ceil(
            points[:, 1].max()
        )

        new_width = int(
            max_x - min_x
        )

        new_height = int(
            max_y - min_y
        )

        if (
            new_width <= 0
            or new_height <= 0
        ):
            raise RuntimeError(
                "Invalid output canvas"
            )

        # -----------------------------------------
        # TRANSLATION
        # -----------------------------------------

        T = np.array(
            [
                [1.0, 0.0, -min_x],
                [0.0, 1.0, -min_y],
                [0.0, 0.0, 1.0]
            ],
            dtype=np.float64
        )

        final_H = T @ H

        # -----------------------------------------
        # WARP
        # -----------------------------------------

        warped = cv2.warpPerspective(
            image,
            final_H,
            (
                new_width,
                new_height
            ),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # -----------------------------------------
        # CREATE VALID AREA MASK
        # -----------------------------------------

        mask = np.ones(
            (height, width),
            dtype=np.uint8
        ) * 255

        warped_mask = cv2.warpPerspective(
            mask,
            final_H,
            (
                new_width,
                new_height
            ),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # -----------------------------------------
        # FIND VALID CONTENT AREA
        # -----------------------------------------

        coordinates = cv2.findNonZero(
            warped_mask
        )

        if coordinates is None:
            raise RuntimeError(
                "No valid image pixels after warp"
            )

        x, y, w, h = cv2.boundingRect(
            coordinates
        )

        cropped = warped[
            y:y + h,
            x:x + w
        ]

        # -----------------------------------------
        # RESIZE BACK TO ORIGINAL SIZE
        # -----------------------------------------

        final_image = cv2.resize(
            cropped,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        print(
            "\n===== FINAL OUTPUT ====="
        )

        print(
            f"Original size: "
            f"{width} x {height}"
        )

        print(
            f"Final size: "
            f"{width} x {height}"
        )

        return final_image