"""
@author: Fish
@title: FishTools
@nickname: FishTools
@description: This extension provides tools for generating laser cutter ready files and other fun stuff
"""
from .LaserCutterFull import LaserCutterFull
from .Deptherize import Deptherize
from .AnaglyphCreator import AnaglyphCreator

# I copied this init from LtCmData
version_str = "0.2"
print(f"### Loading: FishTools ({version_str})")

node_list = [
    "LaserCutterFull",
    "Deptherize",
    "AnaglyphCreator"
]

NODE_CLASS_MAPPINGS = {
    "LaserCutterFull": LaserCutterFull,
    "Deptherize" : Deptherize,
    "AnaglyphCreator" : AnaglyphCreator
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LaserCutterFull": "LaserCutterFull",
    "Deptherize" : "Deptherize",
    "AnaglyphCreator": "AnaglyphCreator"
    }