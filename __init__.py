"""
@author: Fish
@title: FishTools
@nickname: FishTools
@description: This extension provides tools for generating laser cutter ready files
"""
from .LaserCutterFull import LaserCutterFull
from .Deptherize import Deptherize

# I copied this init from LtCmData
version_str = "0.2"
print(f"### Loading: FishTools ({version_str})")

node_list = [
    "LaserCutterFull",
    "Deptherize"
]

NODE_CLASS_MAPPINGS = {
    "LaserCutterFull": LaserCutterFull,
    "Deptherize" : Deptherize
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LaserCutterFull": "LaserCutterFull",
    "Deptherize" : "Deptherize"
    }