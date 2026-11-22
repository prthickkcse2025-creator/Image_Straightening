import os
import argparse
import cv2
import numpy as np

from straightener.detector import VanishingPointDetector
from straightener.geometry import GeometryRectifier

# Cropper is optional
try:
    from straightener.cropper import SmartCropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False


# ==========================================================
# COMMAND-LINE INPUT / OUTPUT
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-estate architectural image straightener"
    )
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--output", required=True, help="Path to save the straightened image")
    return parser.parse_args()


# ==========================================================
# SAFE BLACK BORDER CROPPER
# ==========================================================

def safe_crop_black_borders(
    image,
    threshold=10,
    min_crop_ratio=0.50
):
    """
    Safely removes black borders.

    If detection is unreliable or the crop becomes too small,
    the original rectified image is returned.
    """

    print("\n===== SAFE BORDER CROP =====")

    if image is None:
        print("WARNING: Empty image. Skipping crop.")
        return image

    height, width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Valid pixels are brighter than threshold
    mask = gray > threshold

    # Find valid rows and columns
    rows = np.where(
        np.any(mask, axis=1)
    )[0]

    cols = np.where(
        np.any(mask, axis=0)
    )[0]

    if len(rows) == 0 or len(cols) == 0:
        print(
            "WARNING: No valid image area detected."
        )
        print(
            "Keeping original rectified image."
        )
        return image

    top = int(rows[0])
    bottom = int(rows[-1]) + 1

    left = int(cols[0])
    right = int(cols[-1]) + 1

    crop_width = right - left
    crop_height = bottom - top

    print(
        f"Detected crop area: "
        f"{crop_width} x {crop_height}"
    )

    # Safety check
    if (
        crop_width < width * min_crop_ratio
        or
        crop_height < height * min_crop_ratio
    ):
        print(
            "WARNING: Crop became too small."
        )
        print(
            "Border detection is unreliable."
        )
        print(
            "Keeping full rectified image."
        )

        return image

    cropped = image[
        top:bottom,
        left:right
    ]

    print(
        f"Crop successful: "
        f"{cropped.shape[1]} x "
        f"{cropped.shape[0]}"
    )

    return cropped


# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()
    IMAGE_PATH = args.input
    OUTPUT_PATH = args.output

    # ======================================================
    # CREATE OUTPUT DIRECTORY
    # ======================================================

    output_directory = os.path.dirname(os.path.abspath(OUTPUT_PATH))
    os.makedirs(output_directory, exist_ok=True)

    # ======================================================
    # LOAD IMAGE
    # ======================================================

    print(
        "\n========================================"
    )

    print(
        "IMAGE STRAIGHTENER"
    )

    print(
        "VERTICAL + HORIZONTAL RECTIFICATION"
    )

    print(
        "========================================"
    )

    print(
        "\nLoading image..."
    )

    image = cv2.imread(
        IMAGE_PATH,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"\nCould not load image:\n"
            f"{IMAGE_PATH}"
        )

    height, width = image.shape[:2]

    print(
        f"\nOriginal width: {width}"
    )

    print(
        f"Original height: {height}"
    )

    # ======================================================
    # FOCAL LENGTH
    # ======================================================

    focal_length = max(
        width,
        height
    )

    print(
        f"\nEstimated focal length: "
        f"{focal_length}"
    )

    # ======================================================
    # VANISHING POINT DETECTION
    # ======================================================

    print(
        "\nDetecting vanishing points..."
    )

    detector = VanishingPointDetector(
        focal_length=focal_length,
        seed=1337,
        line_length_threshold=60
    )

    result = detector.detect(
        IMAGE_PATH
    )

    vertical_vp = result[
        "vertical_vp"
    ]

    horizontal_vp_1 = result[
        "horizontal_vp_1"
    ]

    horizontal_vp_2 = result[
        "horizontal_vp_2"
    ]

    print(
        "\n===== DETECTED DIRECTIONS ====="
    )

    print(
        "\nVertical VP:"
    )

    print(
        vertical_vp
    )

    print(
        "\nHorizontal VP 1:"
    )

    print(
        horizontal_vp_1
    )

    print(
        "\nHorizontal VP 2:"
    )

    print(
        horizontal_vp_2
    )

    # ======================================================
    # BUILD RECTIFICATION
    # ======================================================

    print(
        "\nBuilding rectification transformation..."
    )

    rectifier = GeometryRectifier(
        focal_length=focal_length
    )

    H, R, K = (
        rectifier.build_rectification_homography(
            vertical_vp=vertical_vp,
            horizontal_vp_1=horizontal_vp_1,
            horizontal_vp_2=horizontal_vp_2,
            width=width,
            height=height
        )
    )

    # ======================================================
    # PRINT CAMERA MATRIX
    # ======================================================

    print(
        "\n===== CAMERA INTRINSIC MATRIX ====="
    )

    print(
        K
    )

    # ======================================================
    # PRINT ROTATION
    # ======================================================

    print(
        "\n===== ROTATION MATRIX ====="
    )

    print(
        R
    )

    # ======================================================
    # PRINT HOMOGRAPHY
    # ======================================================

    print(
        "\n===== RECTIFICATION HOMOGRAPHY ====="
    )

    print(
        H
    )

    # ======================================================
    # APPLY RECTIFICATION
    # ======================================================

    print(
        "\nStraightening image..."
    )

    try:

        rectified = rectifier.warp_image(
            image=image,
            H=H
        )

    except Exception as error:

        print(
            "\nWARNING: Rectification failed."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "\nUsing original image instead."
        )

        rectified = image.copy()

    # ======================================================
    # VALIDATE OUTPUT
    # ======================================================

    if rectified is None:

        raise RuntimeError(
            "\nRectification produced no image."
        )

    final_height, final_width = (
        rectified.shape[:2]
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
        f"{final_width} x {final_height}"
    )

    # ======================================================
    # BORDER REMOVAL
    # ======================================================

    print(
        "\n===== RECTIFIED OUTPUT ====="
    )

    print(
        f"Size before crop: "
        f"{final_width} x {final_height}"
    )

    print(
        "\nRemoving black borders..."
    )

    # ------------------------------------------------------
    # First try SmartCropper
    # ------------------------------------------------------

    crop_successful = False

    if CROPPER_AVAILABLE:

        try:

            print(
                "\nTrying SmartCropper..."
            )

            # Use only arguments supported by your cropper
            cropper = SmartCropper()

            cropped = cropper.crop(
                rectified
            )

            # Validate result
            if cropped is not None:

                crop_height, crop_width = (
                    cropped.shape[:2]
                )

                # Prevent extremely small / bad crop
                if (
                    crop_width >= width * 0.50
                    and
                    crop_height >= height * 0.50
                ):

                    rectified = cropped

                    crop_successful = True

                    print(
                        "SmartCropper successful."
                    )

                else:

                    print(
                        "WARNING: SmartCropper result "
                        "was too small."
                    )

                    print(
                        "Ignoring SmartCropper result."
                    )

        except Exception as error:

            print(
                "\nWARNING: SmartCropper failed."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "Switching to safe crop."
            )

    # ------------------------------------------------------
    # Safe fallback crop
    # ------------------------------------------------------

    if not crop_successful:

        rectified = safe_crop_black_borders(
            image=rectified,
            threshold=8,
            min_crop_ratio=0.50
        )

    # ======================================================
    # FINAL SIZE
    # ======================================================

    final_height, final_width = (
        rectified.shape[:2]
    )

    print(
        "\n===== FINAL IMAGE SIZE ====="
    )

    print(
        f"Width: {final_width}"
    )

    print(
        f"Height: {final_height}"
    )

    # ======================================================
    # SAVE IMAGE
    # ======================================================

    print(
        "\nSaving image..."
    )

    success = cv2.imwrite(
        OUTPUT_PATH,
        rectified,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )

    if not success:

        raise RuntimeError(
            f"\nFailed to save image:\n"
            f"{OUTPUT_PATH}"
        )

    # ======================================================
    # COMPLETE
    # ======================================================

    print(
        "\n========================================"
    )

    print(
        "VERTICAL + HORIZONTAL "
        "STRAIGHTENING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nInput:"
    )

    print(
        IMAGE_PATH
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nDone!\n"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()