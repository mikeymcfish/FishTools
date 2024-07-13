"""
@author: Fish
@title: FishTools
@nickname: FishTools
@description: This extension provides tools for generating laser cutter ready files and other fun stuff
"""
from .LaserCutterFull import LaserCutterFull
from .AnaglyphCreator import AnaglyphCreator
from .AnaglyphCreatorPro import AnaglyphCreatorPro
from .ShadowMap import ShadowMap

# I copied this init from LtCmData
version_str = "0.3"
print(f"### Loading: FishTools ({version_str})")

node_list = [
    "LaserCutterFull",
    "AnaglyphCreator",
    "AnaglyphCreatorPro",
    "ShadowMap"
]

NODE_CLASS_MAPPINGS = {
    "LaserCutterFull": LaserCutterFull,
    "AnaglyphCreator" : AnaglyphCreator,
    "AnaglyphCreatorPro" : AnaglyphCreatorPro,
    "ShadowMap" : ShadowMap
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LaserCutterFull": "LaserCutterFull",
    "AnaglyphCreator": "AnaglyphCreator",
    "AnaglyphCreatorPro" : "AnaglyphCreatorPro",
    "ShadowMap" : "Shadow Map"
    }