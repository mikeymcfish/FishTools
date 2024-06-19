"""
@author: Fish
@title: FishTools
@nickname: FishTools
@description: This extension provides tools for generating laser cutter ready files
"""
from .LaserCutterFull import LaserCutterFull
from .Deptherize import Deptherize
from .simplifySVG import SimplifySVG

# I copied this init from LtCmData
version_str = "0.1"
print(f"### Loading: FishTools ({version_str})")

node_list = [
    "LaserCutterFull",
    "Deptherize",
    "SimplifySVG"
]

NODE_CLASS_MAPPINGS = {
    "LaserCutterFull": LaserCutterFull,
    "Deptherize" : Deptherize,
    "SimplifySVG": SimplifySVG
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LaserCutterFull": "LaserCutterFull",
    "Deptherize" : "Deptherize",
    "SimplifySVG" : "SimplifySVG"
    }