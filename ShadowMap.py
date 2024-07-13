import cv2
import numpy as np
import argparse
import torch

class ShadowMap:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "azimuth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1}),
                "elevation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "ImageProcessing"

    def run(self, normal_map, azimuth, elevation):
       
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        x = np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = -np.cos(elevation_rad) * np.sin(azimuth_rad)  # Negative to match image coordinates
        z = np.sin(elevation_rad)
        light_dir = np.array([x, y, z], dtype=np.float32)

        normal_map = (normal_map[0].cpu().numpy() * 255).astype(np.float32)

        # Convert normal map to -1 to 1 range, correcting for the standard normal map encoding
        normal_map = (normal_map / 255.0) * 2.0 - 1.0
        normal_map[..., 2] = -normal_map[..., 2]  # Flip Z component

        # Normalize the normal vectors (they should already be normalized, but just in case)
        normal_map = normal_map / np.linalg.norm(normal_map, axis=2, keepdims=True)

        # Calculate dot product
        shading_map = np.dot(normal_map, light_dir)

        # Clamp values between 0 and 1
        shading_map = np.clip(shading_map, 0, 1)

        # Convert to grayscale
        shading_map = (shading_map * 255).astype(np.uint8)  

        # Convert result to torch tensor
        result_tensor = torch.from_numpy(shading_map).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension

        return (result_tensor,)

NODE_CLASS_MAPPINGS = {
    "ShadowMap": ShadowMap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShadowMap": "Shadow Map"
}