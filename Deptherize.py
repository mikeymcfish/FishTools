import cv2
import numpy as np
import torch
import re

class Deptherize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_data": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth", "debug_info")

    FUNCTION = "run"
    CATEGORY = "FishTools"

    def run(self, svg_data):
        debug_info = []

        # Parse the svg_data to extract layer information
        layer_contours = self.parse_svg_data(svg_data, debug_info)
        debug_info.append(f"Parsed layer contours: {layer_contours}")

        # Initialize the image size and the background color
        image_size = 1024
        background_color = (0, 0, 0)
        depth_image = np.full((image_size, image_size, 3), background_color, dtype=np.uint8)
        debug_info.append(f"Initialized depth image of size {depth_image.shape}")

        # Determine the gray shades for each layer starting from 180 to 90
        num_layers = len(layer_contours)
        if num_layers > 1:
            gray_shades = [90 + int((180 - 90) * (i / (num_layers - 1))) for i in range(num_layers)]
            gray_shades.reverse()  # Reverse to have the first layer as the brightest
        else:
            gray_shades = [135]  # Default to middle gray if there's only one layer
        debug_info.append(f"Gray shades: {gray_shades}")

        # Draw the filled shapes for each layer, with the brightest layer on top
        for layer_index in reversed(range(num_layers)):
            gray_value = gray_shades[layer_index]
            fill_color = (gray_value, gray_value, gray_value)
            debug_info.append(f"Layer {layer_index} with fill color {fill_color}")
            for contour in layer_contours[layer_index]:
                if len(contour) > 2:  # Ensure the contour has enough points to form a shape
                    points = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(depth_image, [points], fill_color)
                    debug_info.append(f"Filled contour in layer {layer_index}")

        # Convert to the required output format
        depth_image = torch.tensor(depth_image).unsqueeze(0).float() / 255.0
        debug_info.append(f"Depth image shape: {depth_image.shape}")

        return (depth_image, "\n".join(debug_info))

    def parse_svg_data(self, svg_data, debug_info):
        # Parse the svg_data to extract contours for each layer
        # This assumes that svg_data contains SVG-like path data
        layer_contours = []
        layer_pattern = re.compile(r'<g id="layer_(\d+)".*?>(.*?)</g>', re.DOTALL)
        path_pattern = re.compile(r'<polyline[^>]*points="([^"]+)"[^>]*>', re.DOTALL)

        layers = layer_pattern.findall(svg_data)
        debug_info.append(f"Found {len(layers)} layers")
        
        for layer_index, layer_content in layers:
            contours = []
            paths = path_pattern.findall(layer_content)
            debug_info.append(f"Layer {layer_index} contains {len(paths)} paths")
            for path in paths:
                points = []
                for point in path.split():
                    x, y = map(float, point.split(','))
                    points.append((x, y))
                if points:
                    contours.append(points)
            layer_contours.append(contours)

        return layer_contours

NODE_CLASS_MAPPINGS = {
    "Deptherize": Deptherize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Deptherize": "Deptherize"
}
