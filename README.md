
# LaserCutterFull and Deptherize Nodes

## Overview

This repository contains two custom nodes, `LaserCutterFull` and `Deptherize`, designed for use in image processing workflows. The `LaserCutterFull` node processes input images to generate layers for laser cutting, while the `Deptherize` node converts SVG data into a depth map image.

## LaserCutterFull

### Description

The `LaserCutterFull` node processes an outline image and a depth map to generate multiple layers for laser cutting. Each layer is assigned based on the intensity values from the depth map, and an additional base layer can be specified.

### Inputs

- `outlines`: The outline image (in grayscale). This should be simple line art.
- `depthmap`: The depth map image (in grayscale). I use depthanything with a mask to filter out background noise.
- `base_layer`: The base layer image (in grayscale). This is the bottom of the laser cutter art. I use the mask from depthanything.
- `num_divisions`: The number of layers to divide the depth map into. 
- `use_approximation`: Boolean to toggle the use of contour approximation.
- `approximation_epsilon`: Epsilon value for the contour approximation.
- `shape_similarity_threshold`: Threshold for filtering nearly identical shapes.

### Outputs

- `layer0` to `layer5`: The six layers generated for laser cutting.
- `base_layer`: The base layer image.
- `combined_svg`: The combined SVG string of all layers. *Note* You will need to save this file with a .svg extension. In the example I am doing this with a simple Save Text node
- `debug_info`: Debug information string.

### Variables

- `num_divisions`: Integer specifying the number of depth layers to generate. Minimum is 2, maximum is 6, default is 6. 
- `use_approximation`: Boolean to enable or disable the use of contour approximation.
- `approximation_epsilon`: Float specifying the epsilon value for contour approximation. Default is 0.01, range is 0.001 to 1.0.
- `shape_similarity_threshold`: Float specifying the threshold for filtering nearly identical shapes. Default is 0.01, range is 0.0 to 1.0.

### Tips

- Try different values for num_divisions. I found good results with 4, 5, and 6
- use-approximation is kind of garbage
- shape-similarity is useful if your image has a lot of fine details you want to filter out. Something around 200 has been good for me but try out different values
- The final SVG is very complex with jagged lines (I'm working on this). If you bring it into Illustrator a simple path smooth fixes this.

## Deptherize

### NOTE: 

The current version of LaserCutterFull does not work with Deptherize (which is really only for making a fake preview anyway)

### Description

The `Deptherize` node converts SVG data into a depth map image. Each layer in the SVG data is assigned a grayscale value to represent depth.

### Inputs

- `svg_data`: The SVG data as a string.

### Outputs

- `depth`: The generated depth map image.
- `debug_info`: Debug information string.

### Variables

- `svg_data`: String containing the SVG data.

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Requirements

The required packages are listed in the `requirements.txt` file:

```
torch
numpy
opencv-python-headless
svgwrite
scikit-image
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
