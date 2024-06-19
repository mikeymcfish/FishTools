import re
import svgwrite
from svgpathtools import parse_path, Path, Line, wsvg

class SimplifySVG:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_data": ("STRING", {}),
                "tolerance": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("simplified_svg", "debug_info")

    FUNCTION = "run"
    CATEGORY = "FishTools"

    def run(self, svg_data, tolerance):
        debug_info = []
        simplified_svg = self.simplify_svg(svg_data, tolerance, debug_info)
        debug_info_str = "\n".join(debug_info)
        return simplified_svg, debug_info_str

    def simplify_svg(self, svg_data, tolerance, debug_info):
        path_pattern = re.compile(r'<path[^>]*d="([^"]+)"[^>]*>', re.IGNORECASE)
        polyline_pattern = re.compile(r'<polyline[^>]*points="([^"]+)"[^>]*>', re.IGNORECASE)

        paths = path_pattern.findall(svg_data)
        polylines = polyline_pattern.findall(svg_data)

        debug_info.append(f"Found {len(paths)} paths and {len(polylines)} polylines")

        simplified_paths = []
        original_colors = []

        for match in path_pattern.finditer(svg_data):
            path = match.group(1)
            style_match = re.search(r'stroke="([^"]+)"', match.group(0))
            color = style_match.group(1) if style_match else "black"
            original_colors.append(color)

            debug_info.append(f"Original path: {path}")
            try:
                parsed_path = parse_path(path)
                simplified_path = self.simplify_path(parsed_path)
                simplified_paths.append(simplified_path.d())
                debug_info.append(f"Simplified path: {simplified_path.d()}")
            except Exception as e:
                debug_info.append(f"Error simplifying path: {e}")

        for match in polyline_pattern.finditer(svg_data):
            points = match.group(1)
            style_match = re.search(r'stroke="([^"]+)"', match.group(0))
            color = style_match.group(1) if style_match else "black"
            original_colors.append(color)

            debug_info.append(f"Original polyline points: {points}")
            try:
                points_list = points.strip().split()
                if len(points_list) < 2:
                    continue

                polyline_path = Path()
                complex_points = [complex(*map(float, pt.split(','))) for pt in points_list]

                # Ensure the path starts correctly
                polyline_path.append(Line(start=complex_points[0], end=complex_points[0]))
                for i in range(1, len(complex_points) - 1, 2):
                    polyline_path.append(Line(start=polyline_path[-1].end, end=complex_points[i]))

                # Ensure the last point is included
                polyline_path.append(Line(start=polyline_path[-1].end, end=complex_points[-1]))

                simplified_polyline = self.simplify_path(polyline_path)
                simplified_paths.append(simplified_polyline.d())
                debug_info.append(f"Simplified polyline: {simplified_polyline.d()}")
            except Exception as e:
                debug_info.append(f"Error simplifying polyline: {e}")

        # Create a new SVG with the simplified paths
        svg_document = svgwrite.Drawing(size=("1024px", "1024px"))
        for path, color in zip(simplified_paths, original_colors):
            svg_document.add(svg_document.path(d=path, fill='none', stroke=color))

        return svg_document.tostring()

    def simplify_path(self, path):
        simplified_path = Path()
        for i, segment in enumerate(path):
            if i % 2 == 0 or i == len(path) - 1:
                simplified_path.append(segment)
        return simplified_path

NODE_CLASS_MAPPINGS = {
    "SimplifySVG": SimplifySVG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplifySVG": "SimplifySVG"
}
