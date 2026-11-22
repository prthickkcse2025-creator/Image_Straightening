# Image Straightening for Real Estate Photography

A computer-vision project for correcting architectural perspective distortion in real-estate photographs using vanishing-point detection and geometric rectification.

## Features

- Vanishing-point detection
- Architectural perspective rectification
- Camera geometry analysis
- Homography-based image warping
- Black-border removal
- Safe cropping
- Command-line interface

## Workflow

```mermaid
flowchart TD
    A[Input Image] --> B[Vanishing Point Detection]
    B --> C[Camera Geometry Analysis]
    C --> D[Rectification Transform]
    D --> E[Perspective Warping]
    E --> F[Border Removal and Crop]
    F --> G[Output Image]

Then press **Enter**.

If the README command is still open, type:

```text
