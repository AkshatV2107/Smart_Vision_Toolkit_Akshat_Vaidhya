# Architecture

Input image
    |
    v
Loader -> Grayscale -> Gaussian Blur -> Canny Edges
    |                         |
    |                         v
    +--------------------> Threshold
                              |
                              v
                       Contour Analysis
                              |
                              v
                    Annotated / Comparison
