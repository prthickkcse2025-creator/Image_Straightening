import cv2
import numpy as np


class PrecisionLeveler:
    """
    Final precision leveling stage.

    Detects dominant horizontal architectural lines using
    Canny + Hough Transform and applies a small rotation
    correction.
    """

    def __init__(
        self,
        max_correction_degrees=8.0
    ):
        self.max_correction_degrees = (
            max_correction_degrees
        )


    def estimate_rotation(
        self,
        image
    ):
        """
        Estimate remaining image tilt using dominant
        horizontal architectural lines.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Slight blur removes small texture noise
        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Detect edges
        edges = cv2.Canny(
            gray,
            50,
            150,
            apertureSize=3
        )

        height, width = gray.shape[:2]

        min_line_length = int(
            width * 0.10
        )

        max_line_gap = int(
            width * 0.02
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        if lines is None:

            print(
                "No reliable horizontal lines detected."
            )

            return 0.0

        angles = []
        weights = []

        for line in lines:

            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            length = np.sqrt(
                dx * dx +
                dy * dy
            )

            if length < min_line_length:
                continue

            angle = np.degrees(
                np.arctan2(dy, dx)
            )

            # Convert to range -90 to +90
            if angle > 90:
                angle -= 180

            if angle < -90:
                angle += 180

            # Only nearly-horizontal lines.
            # Ignore vertical furniture edges.
            if abs(angle) <= 20:

                angles.append(angle)
                weights.append(length)

        if len(angles) < 3:

            print(
                "Not enough reliable horizontal lines."
            )

            return 0.0

        angles = np.array(
            angles,
            dtype=np.float64
        )

        weights = np.array(
            weights,
            dtype=np.float64
        )

        # Robust median removes outlier lines
        median_angle = np.median(
            angles
        )

        # Keep only angles close to median
        valid = np.abs(
            angles - median_angle
        ) < 5.0

        angles = angles[valid]
        weights = weights[valid]

        if len(angles) == 0:
            return 0.0

        # Weighted dominant angle
        dominant_angle = np.average(
            angles,
            weights=weights
        )

        correction = -dominant_angle

        # Safety clamp
        correction = np.clip(
            correction,
            -self.max_correction_degrees,
            self.max_correction_degrees
        )

        print(
            f"Detected dominant tilt: "
            f"{dominant_angle:.4f} degrees"
        )

        print(
            f"Final rotation correction: "
            f"{correction:.4f} degrees"
        )

        return correction


    def rotate_and_crop(
        self,
        image,
        angle
    ):
        """
        Rotate image and crop black borders.
        """

        if abs(angle) < 0.01:

            print(
                "Image already level. "
                "No final rotation needed."
            )

            return image

        height, width = image.shape[:2]

        center = (
            width / 2,
            height / 2
        )

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        # Calculate expanded canvas
        cos = abs(
            rotation_matrix[0, 0]
        )

        sin = abs(
            rotation_matrix[0, 1]
        )

        new_width = int(
            height * sin +
            width * cos
        )

        new_height = int(
            height * cos +
            width * sin
        )

        # Move image to center
        rotation_matrix[0, 2] += (
            new_width / 2 -
            center[0]
        )

        rotation_matrix[1, 2] += (
            new_height / 2 -
            center[1]
        )

        # Rotate
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (
                new_width,
                new_height
            ),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Create original valid-area mask
        mask = np.ones(
            (height, width),
            dtype=np.uint8
        ) * 255

        rotated_mask = cv2.warpAffine(
            mask,
            rotation_matrix,
            (
                new_width,
                new_height
            ),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Find valid pixels
        coordinates = cv2.findNonZero(
            rotated_mask
        )

        if coordinates is None:
            return rotated

        x, y, w, h = cv2.boundingRect(
            coordinates
        )

        cropped = rotated[
            y:y + h,
            x:x + w
        ]

        # Resize back to original dimensions
        final_image = cv2.resize(
            cropped,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        return final_image


    def level(
        self,
        image
    ):
        """
        Complete precision leveling pipeline.
        """

        print(
            "\n===== PRECISION LEVELING ====="
        )

        angle = self.estimate_rotation(
            image
        )

        result = self.rotate_and_crop(
            image,
            angle
        )

        print(
            "Precision leveling complete."
        )

        return result