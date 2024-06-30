import cv2
import numpy as np
import torch

## Shout out and thanks to https://github.com/thygate for the inspiration to the new methods!

class AnaglyphCreatorPro:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depthmap": ("IMAGE",),
                "divergence": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "separation": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "stereo_balance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "stereo_offset_exponent": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "fill_technique": (["naive", "naive_interpolating", "polylines_soft", "polylines_sharp"],),
                "output_mode": (["red-cyan-anaglyph", "left-right", "top-bottom"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "ImageProcessing"

    def run(self, image, depthmap, divergence, separation, stereo_balance, stereo_offset_exponent, fill_technique, output_mode):
        # Convert inputs to NumPy arrays
        image = (image[0].cpu().numpy() * 255).astype(np.uint8)
        depthmap = (depthmap[0].cpu().numpy() * 255).astype(np.uint8)

        # Ensure depthmap is single channel
        if len(depthmap.shape) == 3:
            depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)

        # Invert the depth map
        depthmap = 255 - depthmap

        # Normalize depth and calculate divergence in pixels
        normalized_depth = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
        divergence_px = (divergence / 100.0) * image.shape[1]
        separation_px = (separation / 100.0) * image.shape[1]

        # Create left and right images
        balance = (stereo_balance + 1) / 2
        left_eye = self.apply_stereo_divergence(image, normalized_depth, +1 * divergence_px * balance, -1 * separation_px,
                                                stereo_offset_exponent, fill_technique)
        right_eye = self.apply_stereo_divergence(image, normalized_depth, -1 * divergence_px * (1 - balance), separation_px,
                                                 stereo_offset_exponent, fill_technique)

        # Generate output based on selected mode
        if output_mode == "red-cyan-anaglyph":
            result = self.overlap_red_cyan(left_eye, right_eye)
        elif output_mode == "left-right":
            result = np.hstack([left_eye, right_eye])
        elif output_mode == "top-bottom":
            result = np.vstack([left_eye, right_eye])

        # Convert result to torch tensor
        result_tensor = torch.from_numpy(result).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension

        return (result_tensor,)

    @staticmethod
    def apply_stereo_divergence(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique):
        h, w, c = original_image.shape
        derived_image = np.zeros_like(original_image)

        for row in range(h):
            for col in range(w):
                col_d = int(col + (normalized_depth[row][col] ** stereo_offset_exponent) * divergence_px + separation_px)
                if 0 <= col_d < w:
                    derived_image[row][col_d] = original_image[row][col]

        if fill_technique == "naive":
            return AnaglyphCreatorPro.fill_naive(derived_image, divergence_px)
        elif fill_technique == "naive_interpolating":
            return AnaglyphCreatorPro.fill_naive_interpolating(derived_image)
        elif fill_technique in ["polylines_soft", "polylines_sharp"]:
            return AnaglyphCreatorPro.fill_polylines(derived_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique)

        return derived_image

    @staticmethod
    def fill_naive(derived_image, divergence_px):
        h, w, c = derived_image.shape
        filled = np.zeros((h, w), dtype=np.uint8)

        for row in range(h):
            for col in range(w):
                if np.sum(derived_image[row, col]) != 0:
                    filled[row, col] = 1

        for row in range(h):
            for col in range(w):
                if filled[row, col] == 0:
                    for offset in range(1, abs(int(divergence_px)) + 2):
                        if col + offset < w and filled[row, col + offset] == 1:
                            derived_image[row, col] = derived_image[row, col + offset]
                            break
                        if col - offset >= 0 and filled[row, col - offset] == 1:
                            derived_image[row, col] = derived_image[row, col - offset]
                            break

        return derived_image

    @staticmethod
    def fill_naive_interpolating(derived_image):
        h, w, c = derived_image.shape

        for row in range(h):
            l_pointer = 0
            while l_pointer < w:
                if np.sum(derived_image[row, l_pointer]) != 0:
                    l_pointer += 1
                    continue

                r_pointer = l_pointer + 1
                while r_pointer < w and np.sum(derived_image[row, r_pointer]) == 0:
                    r_pointer += 1

                if r_pointer < w:
                    l_color = derived_image[row, l_pointer - 1] if l_pointer > 0 else derived_image[row, r_pointer]
                    r_color = derived_image[row, r_pointer]
                    
                    for col in range(l_pointer, r_pointer):
                        t = (col - l_pointer) / (r_pointer - l_pointer)
                        derived_image[row, col] = ((1 - t) * l_color + t * r_color).astype(np.uint8)

                l_pointer = r_pointer

        return derived_image

    @staticmethod
    def fill_polylines(derived_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique):
        h, w, c = derived_image.shape
        pixel_half_width = 0.45 if fill_technique == "polylines_sharp" else 0.0

        for row in range(h):
            points = []
            for col in range(w):
                coord_d = (normalized_depth[row, col] ** stereo_offset_exponent) * divergence_px
                coord_x = col + 0.5 + coord_d + separation_px
                if pixel_half_width < 1e-7:
                    points.append((coord_x, abs(coord_d), col))
                else:
                    points.append((coord_x - pixel_half_width, abs(coord_d), col))
                    points.append((coord_x + pixel_half_width, abs(coord_d), col))

            points.sort(key=lambda x: x[0])

            for col in range(w):
                color = np.zeros(3, dtype=np.float32)
                total_weight = 0.0

                for i in range(len(points) - 1):
                    if points[i][0] <= col < points[i+1][0]:
                        weight = 1.0 - abs(col - points[i][0]) / (points[i+1][0] - points[i][0])
                        color += weight * derived_image[row, int(points[i][2])]
                        total_weight += weight

                if total_weight > 0:
                    derived_image[row, col] = (color / total_weight).astype(np.uint8)

        return derived_image

    @staticmethod
    def overlap_red_cyan(im1, im2):
        height, width = im1.shape[:2]
        composite = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            for j in range(width):
                composite[i, j, 0] = im2[i, j, 0]  # Red channel from right image
                composite[i, j, 1:] = im1[i, j, 1:]  # Green and Blue channels from left image

        return composite

NODE_CLASS_MAPPINGS = {
    "AnaglyphCreatorPro": AnaglyphCreatorPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnaglyphCreatorPro": "Anaglyph Creator Pro"
}