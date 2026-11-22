import cv2
import numpy as np
from lu_vp_detect import VPDetection


class VanishingPointDetector:
    """
    Detect and classify vanishing point directions.

    XiaohuLuVPDetection returns normalized 3D direction
    vectors in camera coordinates.
    """

    def __init__(
        self,
        focal_length=None,
        seed=1337,
        line_length_threshold=60
    ):
        self.focal_length = focal_length
        self.seed = seed
        self.line_length_threshold = line_length_threshold

    @staticmethod
    def _normalize(vector):
        """
        Normalize a 3D vector.
        """

        vector = np.asarray(
            vector,
            dtype=np.float64
        )

        norm = np.linalg.norm(vector)

        if norm < 1e-10:
            raise ValueError(
                "Cannot normalize a zero-length vector"
            )

        return vector / norm

    def detect(self, image_path):
        """
        Detect vanishing points from an image.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        height, width = image.shape[:2]

        focal_length = self.focal_length

        if focal_length is None:
            focal_length = max(width, height)

        # Create vanishing point detector
        vpd = VPDetection(
            length_thresh=self.line_length_threshold,
            principal_point=np.array(
                [
                    width / 2,
                    height / 2
                ],
                dtype=np.float64
            ),
            focal_length=focal_length,
            seed=self.seed
        )

        # Detect 3 vanishing point directions
        vps = vpd.find_vps(image_path)

        return self._classify_vanishing_points(
            vps=vps,
            width=width,
            height=height,
            focal_length=focal_length
        )

    def _classify_vanishing_points(
        self,
        vps,
        width,
        height,
        focal_length
    ):
        """
        Classify the three 3D direction vectors.

        Camera coordinates:

            X -> image horizontal direction
            Y -> image vertical direction
            Z -> camera forward/depth direction

        Therefore, the architectural vertical direction
        is the vector with the largest absolute Y component.
        """

        vps = np.asarray(
            vps,
            dtype=np.float64
        )

        if vps.shape != (3, 3):
            raise ValueError(
                f"Expected shape (3, 3), got {vps.shape}"
            )

        # Normalize all directions
        normalized_vps = np.array(
            [
                self._normalize(vp)
                for vp in vps
            ]
        )

        # IMPORTANT:
        # Select the vector most aligned with the Y axis.
        vertical_scores = np.abs(
            normalized_vps[:, 1]
        )

        vertical_index = int(
            np.argmax(vertical_scores)
        )

        vertical_vp = normalized_vps[
            vertical_index
        ]

        # The remaining two directions are horizontal
        horizontal_vps = np.delete(
            normalized_vps,
            vertical_index,
            axis=0
        )

        return {
            "vertical_vp": vertical_vp,
            "horizontal_vp_1": horizontal_vps[0],
            "horizontal_vp_2": horizontal_vps[1],
            "all_vps": normalized_vps,
            "image_width": width,
            "image_height": height,
            "focal_length": focal_length
        }