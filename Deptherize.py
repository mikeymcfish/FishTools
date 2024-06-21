import svgwrite
from xml.etree import ElementTree as ET
import torch
from PIL import Image, ImageDraw
import numpy as np
from svgpathtools import parse_path, Path, Line, CubicBezier

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

    def modify_svg(self, svg_string):
        svg_ns = {'svg': 'http://www.w3.org/2000/svg'}
        tree = ET.ElementTree(ET.fromstring(svg_string))
        root = tree.getroot()

        layers = [layer for layer in root.findall('svg:g', svg_ns) if layer.get('id').startswith('layer_')]

        for i, layer in enumerate(layers):
            shade = int(255 * (1 - (i / len(layers))))
            fill_color = f'rgb({shade},{shade},{shade})'
            
            for path in layer.findall('svg:path', svg_ns):
                path.set('stroke', 'none')
                path.set('fill', fill_color)

        modified_svg_string = ET.tostring(root).decode()
        return modified_svg_string

    def svg_to_bitmap(self, svg_string):
        svg_ns = {'svg': 'http://www.w3.org/2000/svg'}
        tree = ET.ElementTree(ET.fromstring(svg_string))
        root = tree.getroot()

        width = 1024
        height = 1024

        image = Image.new('L', (width, height), 255)  # Create a white grayscale image
        draw = ImageDraw.Draw(image)

        layers = [layer for layer in root.findall('svg:g', svg_ns) if layer.get('id').startswith('layer_')]

        for layer in layers:
            for path in layer.findall('svg:path', svg_ns):
                d = path.get('d')
                parsed_path = parse_path(d)
                for segment in parsed_path:
                    if isinstance(segment, Line):
                        draw.line([segment.start.real, segment.start.imag, segment.end.real, segment.end.imag], fill=0)
                    elif isinstance(segment, CubicBezier):
                        draw.bezier([segment.start.real, segment.start.imag,
                                     segment.control1.real, segment.control1.imag,
                                     segment.control2.real, segment.control2.imag,
                                     segment.end.real, segment.end.imag], fill=0)

        return image

    def run(self, svg_data):
        debug_info = []

        # Convert to the required output format
        svg_new = self.modify_svg(svg_data)

        # Convert modified SVG to bitmap
        image = self.svg_to_bitmap(svg_new)

        image_np = np.array(image)  # Convert to numpy array

        # Convert numpy array to PyTorch tensor
        depth_image = torch.tensor(image_np).unsqueeze(0).float() / 255.0

        debug_info.append(f"Depth image shape: {depth_image.shape}")

        return (depth_image, "\n".join(debug_info))

NODE_CLASS_MAPPINGS = {
    "Deptherize": Deptherize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Deptherize": "Deptherize"
}
