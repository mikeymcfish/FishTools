import cv2
import numpy as np
import torch

class AnaglyphCreator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depthmap": ("IMAGE",),
                "anaglyph_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("anaglyph_image", "visualization_image", "overlap_mask", "left_debug_image", "right_debug_image")

    FUNCTION = "run"
    CATEGORY = "ImageProcessing"

    def run(self, image, depthmap, anaglyph_shift):
        # Convert inputs to NumPy arrays
        image = (image[0].cpu().numpy().copy() * 255).astype(np.uint8)
        depthmap = (depthmap[0].cpu().numpy().copy() * 255).astype(np.uint8)

        # Ensure depthmap is single channel
        if len(depthmap.shape) == 3:
            depthmap = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)

         # Invert the depth map
        depthmap = 255 - depthmap

        height, width = image.shape[:2]

        # Create left (cyan) and right (red) images - swapped from previous version
        left_image = np.zeros((height, width, 3), dtype=np.float32)
        right_image = np.zeros((height, width, 3), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                # Calculate shift based on depth, use full shift for left image, no shift for right
                shift = anaglyph_shift * (depthmap[y, x] / 255.0)
                left_x = x - shift
                right_x = x

                # Interpolate for smoother transitions in left image
                left_x_int, left_x_frac = int(left_x), left_x - int(left_x)

                # Left image (cyan channels) - shifted with interpolation
                if 0 <= left_x_int < width - 1:
                    for c in range(1, 3):  # Green and Blue channels
                        left_image[y, x, c] = (image[y, left_x_int, c] * (1 - left_x_frac) + 
                                               image[y, min(left_x_int + 1, width - 1), c] * left_x_frac)

                # Right image (red channel) - no shift
                right_image[y, x, 0] = image[y, x, 0]  # Red channel

                # right_x = x - shift
                # left_x = x

                # # Interpolate for smoother transitions in left image
                # right_x_int, right_x_frac = int(right_x), right_x - int(right_x)

                # # Left image (cyan channels) - shifted with interpolation
                # if 0 <= right_x_int < width - 1:
                #     for c in range(1, 3):  # Green and Blue channels
                #         right_image[y, x, c] = (image[y, right_x_int, c] * (1 - right_x_frac) + 
                #                                image[y, min(right_x_int + 1, width - 1), c] * right_x_frac)

                # # Right image (red channel) - no shift
                # left_image[y, x, 0] = image[y, x, 0]  # Red channel

        # Combine left and right images to create anaglyph
        anaglyph = np.zeros_like(image, dtype=np.float32)
        anaglyph[:,:,1:] = left_image[:,:,1:]  # Green and Blue channels from left image
        anaglyph[:,:,0] = right_image[:,:,0]  # Red channel from right image

        # Apply edge-aware filtering to reduce artifacts
        anaglyph = cv2.edgePreservingFilter(anaglyph.astype(np.uint8), flags=1, sigma_s=60, sigma_r=0.4)

        # Visualization of shifts
        visualization_image = self.visualize_shifts(image, depthmap, anaglyph_shift)

        # Overlap mask
        overlap_mask = self.create_overlap_mask(left_image, right_image)

        # Channel debug images
        left_debug_image = np.zeros_like(image)
        left_debug_image[:,:,1:] = left_image[:,:,1:]  # Green and Blue channels

        right_debug_image = np.zeros_like(image)
        right_debug_image[:,:,0] = right_image[:,:,0]  # Only red channel

        # Convert all outputs to the correct format
        anaglyph_tensor = self.format_output(anaglyph)
        visualization_tensor = self.format_output(visualization_image)
        overlap_tensor = self.format_output(overlap_mask, grayscale=True)

        #Swap because idk that's what it needs
        left_debug_tensor = self.format_output(left_debug_image)
        right_debug_tensor = self.format_output(right_debug_image)

        return anaglyph_tensor, visualization_tensor, overlap_tensor, left_debug_tensor, right_debug_tensor

    def visualize_shifts(self, image, depthmap, anaglyph_shift):
        height, width = image.shape[:2]
        visualization = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                shift = int(anaglyph_shift * (depthmap[y, x] / 255.0))
                new_x = x - shift
                if 0 <= new_x < width:
                    visualization[y, x] = [0, 255, 0]  # Mark the original position in green
                    visualization[y, new_x] = [0, 0, 255]  # Mark the shifted position in red

        return visualization

    def create_overlap_mask(self, left_image, right_image):
        overlap_mask = np.zeros(left_image.shape[:2], dtype=np.uint8)
        overlap_mask[np.any(left_image[:,:,1:] > 0, axis=2)] = 128  # Mark left image pixels
        overlap_mask[right_image[:,:,0] > 0] += 128  # Mark right image pixels
        return overlap_mask

    def format_output(self, image, grayscale=False):
        height, width = image.shape[:2]
        if grayscale:
            output = np.full((height, width), 255, dtype=np.uint8)
        else:
            output = np.full((height, width, 3), 255, dtype=np.uint8)
        output[:height, :width] = image
        return torch.tensor(output).unsqueeze(0).float() / 255.0

NODE_CLASS_MAPPINGS = {
    "AnaglyphCreator": AnaglyphCreator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnaglyphCreator": "AnaglyphCreator"
}