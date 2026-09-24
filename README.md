# grid-puzzle-vision-solver
An automated grid puzzle solver using OpenCV for board and cell detection, K-Means clustering for color classification, and backtracking search for constraint-based puzzle solving.
## Pipeline

The solver processes the puzzle image through the following pipeline:

                         game.jpg
                            │
                            ▼
                    Image Preprocessing
                 Grayscale + Thresholding
                       + Downsampling
                            │
                            ▼
                    Contour Detection
                            │
                            ▼
                 Contour Hierarchy Analysis
                            │
                            ▼
              Geometric Board Identification
              Area + Aspect Ratio + Quadrilateral
                            │
                            ▼
                    Cell Extraction
                         64 Cells
                            │
                            ▼
                 Color Feature Extraction
                   Mean Cell Color (BGR)
                            │
                            ▼
                    K-Means Clustering
                            │
                            ▼
                   Color → Cluster Label
                            │
                            ▼
                 Spatial Grid Reconstruction
                            │
                            ▼
                         8 × 8
                          Matrix
                            │
                            ▼
                  Backtracking Search
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
          Column Constraint  Color     Neighborhood
                            Constraint   Constraint
                │           │           │
                └───────────┼───────────┘
                            ▼
                         Solution

### 1. Image Preprocessing

The input screenshot is downsampled and converted to grayscale, followed by binary thresholding to simplify subsequent contour detection.

### 2. Board Detection

Contours are extracted using OpenCV. The contour hierarchy is analyzed to identify the puzzle board based on its geometric properties, including area, aspect ratio, quadrilateral shape, and the number of child contours.

### 3. Cell Extraction

The child contours of the detected board are treated as individual puzzle cells. For each cell, its center coordinates and mean BGR color are extracted.

### 4. Color Classification

K-Means clustering is applied to the mean cell colors. Each cell is assigned a cluster label representing its color category.

### 5. Grid Reconstruction

The detected cell centers are spatially sorted by their vertical and horizontal coordinates to reconstruct the puzzle as an `N × N` matrix.

### 6. Puzzle Solving

A backtracking search algorithm is used to find a valid flag placement. The solver enforces the following constraints:

* **One flag per row**
* **One flag per column**
* **One flag per color**
* **No flags in adjacent cells within the 8-neighborhood**

The search continues until a valid configuration satisfying all constraints is found.
