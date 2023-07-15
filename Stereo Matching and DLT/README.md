# Stereo Matching and DLT Toolbox

This repository contains a collection of Python functions for implementing stereo matching and DLT (Direct Linear Transform) algorithm for homography estimation and warping. These functions were implemented as part of an assignment to demonstrate proficiency in stereo matching and image warping.

## Functions

### 1. Stereo Matching

#### 1.1 SSD

- `disparitySSD`: Calculates the disparity map between left and right images using Sum of Squared Differences (SSD) method.

#### 1.2 Normalized Correlation

- `disparityNC`: Calculates the disparity map between left and right images using Normalized Correlation method.

### 2. Homography and Warping

#### 2.1 DLT

- `computeHomography`: Calculates the homography matrix using the Direct Linear Transform (DLT) algorithm.

#### 2.2 Warping

- `warpImage`: Warps an image onto another image using a user-defined set of corresponding points.

## Usage

To use the functions in this toolbox, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/stereo-matching-dlt-toolbox.git
   ```

2. Import the necessary functions into your Python script:

   ```python
   from stereo_matching_dlt_toolbox import disparitySSD, disparityNC, computeHomography, warpImage
   ```

3. Call the desired function with the appropriate inputs and utilize the returned values as needed.

## Examples

Here are some examples demonstrating how to use the functions in this toolbox:

```python
# Example usage of disparitySSD
disparity_map_ssd = disparitySSD(img_l, img_r, disp_range, k_size)

# Example usage of disparityNC
disparity_map_nc = disparityNC(img_l, img_r, disp_range, k_size)

# Example usage of computeHomography
homography_matrix, homography_error = computeHomography(src_pnt, dst_pnt)

# Example usage of warpImage
warpImage(src_img, dst_img)
```

Feel free to explore the toolbox and modify the functions to suit your specific needs.
