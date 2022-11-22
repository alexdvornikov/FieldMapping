"""
Set global variables with detector and physics properties
"""
from . import detector

def load_properties(detprop_file, pixel_file):
    """
    The function loads the detector properties and
    the pixel geometry YAML files and stores the constants
    as global variables

    Args:
        detprop_file (str): detector properties YAML filename
        pixel_file (str): pixel layout YAML filename
    """
    detector.set_detector_properties(detprop_file, pixel_file)
