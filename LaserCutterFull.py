import cv2
import numpy as np
import torch
import svgwrite
from skimage import measure
from PIL import Image, ImageDraw
import vtracer
import lxml.etree as ET
import re

class LaserCutterFull:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outlines": ("IMAGE",),
                "depthmap": ("IMAGE",),
                "base_layer": ("IMAGE",),
                "num_divisions": ("INT", {
                    "default": 6, 
                    "min": 2, 
                    "max": 6, 
                    "step": 1, 
                    "display": "number"
                }),
                "use_approximation": ("BOOLEAN", {"default": False}),
                "approximation_epsilon": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "shape_similarity_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "min_shape_area": ("FLOAT", {"default": 250.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "apply_blur": ("BOOLEAN", {"default": False}),
                "corner_threshold" : ("INT", {
                    "default": 60, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1, 
                    "display": "number"
                }),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_iterations" : ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 20, 
                    "step": 1, 
                    "display": "number"
                }),
                "splice_threshold" : ("INT", {
                    "default": 45, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1, 
                    "display": "number"
                }),
                "path_precision" : ("INT", {
                    "default": 3, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1, 
                    "display": "number"
                }),
                          
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("layer0", "layer1", "layer2", "layer3", "layer4", "layer5", "base_layer", "combined_svg", "debug_info")

    FUNCTION = "run"
    CATEGORY = "FishTools"

    def preprocess_image(self, image, to_gray=True, threshold=240):
        if len(image.shape) == 3 and to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        return binary_image

    def extract_contours(self, binary_image, use_approximation=False, approximation_epsilon=0.01):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if use_approximation:
            contours = [cv2.approxPolyDP(contour, approximation_epsilon * cv2.arcLength(contour, True), True) for contour in contours]
        return contours

    def calculate_intensities(self, contours, depthmap, min_shape_area):
        shape_intensity = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < min_shape_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w > (0.9 * 1024) and h > (0.9 * 1024):
                continue
            mask = np.zeros_like(depthmap)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            labels = measure.label(mask, connectivity=2, background=0)
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == 0:
                    continue
                shape_mask = np.zeros_like(depthmap)
                shape_mask[labels == label] = 255
                shape_pixels = depthmap[shape_mask == 255]
                if len(shape_pixels) == 0:
                    continue
                mode_intensity_value = np.bincount(np.ravel(shape_pixels)).argmax()
                shape_intensity.append((i, mode_intensity_value, contour))
        return shape_intensity

    def assign_shapes_to_layers(self, shape_intensity, binary_image, num_divisions):
        combined_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
        layer_images = {}
        layer_contours = {}
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 0, 128), (0, 128, 128), (128, 128, 0), (255, 165, 0)
        ]
        svg_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'purple', 'teal', 'olive', 'orange']
        
        shapes_in_layers = {i: [] for i in range(num_divisions)}
        layer_masks = [np.zeros_like(binary_image) for _ in range(num_divisions)]
        for i, (shape_index, mode_intensity_value, contour) in enumerate(shape_intensity):
            shape_mask = np.zeros_like(binary_image)
            cv2.drawContours(shape_mask, [contour], -1, (255), thickness=cv2.FILLED)
            mask_indices = np.where(shape_mask == 255)

            assigned = False
            for layer_index in range(num_divisions):
                if not np.any(layer_masks[layer_index][mask_indices]):
                    layer_masks[layer_index][mask_indices] = 1
                    color = colors[layer_index % len(colors)]
                    svg_color = svg_colors[layer_index % len(svg_colors)]

                    color_mask = np.zeros_like(combined_image)
                    color_mask[mask_indices] = color
                    combined_image[mask_indices[0], mask_indices[1], :] = color

                    if layer_index not in layer_images:
                        layer_images[layer_index] = np.zeros_like(combined_image)
                        layer_contours[layer_index] = []

                    layer_images[layer_index][mask_indices[0], mask_indices[1], :] = color
                    layer_contours[layer_index].append((contour, svg_color))
                    color_mask_alpha = np.full_like(combined_image, 255)
                    color_mask_alpha[mask_indices[0], mask_indices[1], :] = (0, 0, 0)
                    combined_image[mask_indices[0], mask_indices[1], :] = cv2.addWeighted(
                        combined_image[mask_indices[0], mask_indices[1], :], 1 - 0.25,
                        color_mask_alpha[mask_indices[0], mask_indices[1], :], 0.25, 0)
                    cv2.drawContours(combined_image, [contour], -1, color, 2)

                    shapes_in_layers[layer_index].append(shape_index)
                    assigned = True
                    break

            if not assigned:
                shapes_in_layers[num_divisions - 1].append(shape_index)

        return layer_images, layer_contours, shapes_in_layers

    def contours_to_image(self, contours, shape, debug_info):
        image = Image.new('L', (shape[1], shape[0]), 255)
        draw = ImageDraw.Draw(image)
        
        for contour_info in contours:
            if isinstance(contour_info, tuple):
                contour = contour_info[0]  # Extract the contour if it's a tuple
            else:
                contour = contour_info  # Use the contour directly if it's not a tuple
            try:
                # Attempt to handle different possible structures of the contour points
                points = [(int(point[0][0]), int(point[0][1])) for point in contour]  # Common structure
            except TypeError:
                points = [(int(point[0]), int(point[1])) for point in contour]  # Fallback structure
            # debug_info.append(f"Points list {points}")
            draw.polygon(points, outline=0, width=3)
            
        debug_info.append(f"Converted {len(contours)} contours to image.")
        return image
    

    def convert_to_svg(self, layer_contours, base_layer_contours, shape, debug_info,
                       corner_threshold, length_threshold, max_iterations, splice_threshold, path_precision,
                       colormode="binary", hierarchical="cutout", mode="spline", 
                       filter_speckle=4, color_precision=6, layer_difference=16, 
                      ):
        # Convert layer contours to images
        layer_images = {layer_index: self.contours_to_image(contours, shape, debug_info) for layer_index, contours in layer_contours.items()}
        base_layer_image = self.contours_to_image(base_layer_contours, shape, debug_info)

        svg_strings = []

        # Process each layer
        for layer_index, image in layer_images.items():
            image = image.convert("RGBA")
            pixels = list(image.getdata())

            # Convert the pixels list back to an Image object
            width, height = image.size
            new_image = Image.new("RGBA", (width, height))
            new_image.putdata(pixels)

            size = image.size
            color = layer_contours[layer_index][0][1] if layer_contours[layer_index] else "black"
            svg_str = vtracer.convert_pixels_to_svg(
                pixels,
                size=size,
                colormode=colormode,
                hierarchical=hierarchical,
                mode=mode,
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                layer_difference=layer_difference,
                corner_threshold=corner_threshold,
                length_threshold=length_threshold,
                max_iterations=max_iterations,
                splice_threshold=splice_threshold,
                path_precision=path_precision
            )

             # Perform string manipulations
            svg_str = re.sub(r'(<path[^>]*?) fill="[^"]*"', rf'\1 fill="none" stroke="{color}"', svg_str)
            path_pattern = re.compile(r'(<path[^>]*?d=".*?M.*?)(M.*?)(?=".*?fill=)')
            
            # Function to process each match
            def process_match(match):
                # Keep everything before the second "M" and add fill and stroke attributes
                return f'{match.group(1)}'
            
            # Replace the matches in the SVG string
            svg_str = re.sub(path_pattern, process_match, svg_str)

             # Remove <svg>, </svg>, and <?xml> tags
            svg_str = re.sub(r'<\?xml[^>]*\?>', '', svg_str)  # Remove <?xml ... ?> tags
            svg_str = re.sub(r'<svg[^>]*>', '', svg_str)      # Remove <svg ... > tags
            svg_str = re.sub(r'</svg>', '', svg_str)          # Remove </svg> tags

            svg_strings.append(f'<g id="layer_{layer_index}" data-name="layer_{layer_index}">{svg_str.strip()}</g>')

        # Process the base layer
        base_layer_image = base_layer_image.convert("RGBA")
        pixels = list(base_layer_image.getdata())
        size = base_layer_image.size
        debug_info.append(f"Pixels length: {len(pixels)}")

        base_svg_str = vtracer.convert_pixels_to_svg(
            pixels,
            size=size,
            colormode=colormode,
            hierarchical=hierarchical,
            mode=mode,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            layer_difference=layer_difference,
            corner_threshold=corner_threshold,
            length_threshold=length_threshold,
            max_iterations=max_iterations,
            splice_threshold=splice_threshold,
            path_precision=path_precision
        )

        # Perform string manipulations
        base_svg_str = re.sub(r'(<path[^>]*?) fill="[^"]*"', rf'\1 fill="none" stroke="{color}"', base_svg_str)
        path_pattern = re.compile(r'(<path[^>]*?d=".*?M.*?)(M.*?)(?=".*?fill=)')
        
        # Function to process each match
        def process_match(match):
            # Keep everything before the second "M" and add fill and stroke attributes
            return f'{match.group(1)}'
        
        # Replace the matches in the SVG string
        base_svg_str = re.sub(path_pattern, process_match, base_svg_str)

            # Remove <svg>, </svg>, and <?xml> tags
        base_svg_str = re.sub(r'<\?xml[^>]*\?>', '', base_svg_str)  # Remove <?xml ... ?> tags
        base_svg_str = re.sub(r'<svg[^>]*>', '', base_svg_str)      # Remove <svg ... > tags
        base_svg_str = re.sub(r'</svg>', '', base_svg_str)          # Remove </svg> tags

         # Combine all SVG strings into one document
        combined_svg = '<?xml version="1.0" encoding="UTF-8"?><svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:ev="http://www.w3.org/2001/xml-events" viewBox="0 0 1024 1024">'

        for svg_str in svg_strings:
            combined_svg += svg_str

        combined_svg += f'<g id="layer_base" data-name="layer_base">{base_svg_str.strip()}</g>'
        combined_svg += '</svg>'

        debug_info.append("SVG conversion complete.")

        return combined_svg

    def run(self, outlines, depthmap, base_layer, num_divisions, use_approximation, approximation_epsilon, 
            shape_similarity_threshold, min_shape_area, apply_blur, 
            corner_threshold, length_threshold, max_iterations, splice_threshold, path_precision):
        debug_info = []

        # Section 1: Convert inputs to NumPy arrays
        outlines = (outlines[0].cpu().numpy().copy() * 255).astype(np.uint8)
        depthmap = (depthmap[0].cpu().numpy().copy() * 255).astype(np.uint8)
        base_layer = (base_layer[0].cpu().numpy().copy() * 255).astype(np.uint8)

        if apply_blur:
            depthmap = cv2.GaussianBlur(depthmap, (5, 5), 0)

        # Preprocess images
        outlines_binary = self.preprocess_image(outlines, threshold=200)
        base_layer_binary = self.preprocess_image(base_layer, threshold=200)

        # Extract contours
        contours = self.extract_contours(outlines_binary, use_approximation, approximation_epsilon)
        base_layer_contours = self.extract_contours(base_layer_binary, use_approximation, approximation_epsilon)
        

        # Calculate intensities
        shape_intensity = self.calculate_intensities(contours, depthmap, min_shape_area)

        # Assign shapes to layers
        layer_images, layer_contours, shapes_in_layers = self.assign_shapes_to_layers(shape_intensity, outlines_binary, num_divisions)

        # Convert to SVG
        combined_svg = self.convert_to_svg(layer_contours, base_layer_contours, 
                                           outlines_binary.shape, debug_info, 
                                           corner_threshold, length_threshold, max_iterations, splice_threshold, path_precision)

        # Prepare the output images
        layer_images_result = []
        for layer_index in range(num_divisions):
            if layer_index in layer_images:
                gray_layer_image = cv2.cvtColor(layer_images[layer_index], cv2.COLOR_BGR2GRAY)
                _, binary_layer_image = cv2.threshold(gray_layer_image, 1, 255, cv2.THRESH_BINARY)
                final_contours, _ = cv2.findContours(binary_layer_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                outline_layer_image = np.full_like(outlines_binary, 255)
                cv2.drawContours(outline_layer_image, final_contours, -1, (0), 2)
                outline_layer_image = np.stack([outline_layer_image]*3, axis=-1)
                layer_images_result.append(torch.tensor(outline_layer_image).unsqueeze(0).float() / 255.0)
            else:
                blank_image = np.full_like(outlines_binary, 255)
                blank_image = np.stack([blank_image]*3, axis=-1)
                layer_images_result.append(torch.tensor(blank_image).unsqueeze(0).float() / 255.0)

        while len(layer_images_result) < 6:
            blank_image = np.full_like(outlines_binary, 255)
            blank_image = np.stack([blank_image]*3, axis=-1)
            layer_images_result.append(torch.tensor(blank_image).unsqueeze(0).float() / 255.0)

        debug_info.append(f"Created SVGs")
        # Add base layer output
        base_layer_image = np.full_like(base_layer_binary, 255)
        cv2.drawContours(base_layer_image, base_layer_contours, -1, (0), 2)
        base_layer_image = np.stack([base_layer_image]*3, axis=-1)
        base_layer_image = torch.tensor(base_layer_image).unsqueeze(0).float() / 255.0

        # Debug info
        for layer_index in range(num_divisions):
            debug_info.append(f"Shapes in layer {layer_index}: {shapes_in_layers[layer_index]}")
        debug_info.append(f"Total shapes processed: {len(shape_intensity)}")
        debug_info.append(f"Number of layers returned: {len(layer_images_result)}")
        debug_info_string = "\n".join(debug_info)

        return tuple(layer_images_result[:6]) + (base_layer_image, combined_svg, debug_info_string)

NODE_CLASS_MAPPINGS = {
    "LaserCutterFull": LaserCutterFull
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LaserCutterFull": "LaserCutterFull"
}
