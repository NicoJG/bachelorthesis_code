# Imports
import sys
from pathlib import Path
import json

# Imports from this project
sys.path.insert(0, Path(__file__).parent.parent)
from utils import paths

def load_feature_keys(include_keys, exclude_keys=None):
    """Read in selected features from the features.json

    Args:
        include_keys (list): Which types of features to include (keys of the json file)
        exclude_keys (list, optional): Which types of features to exclude

    Returns:
        list: A concatenated list of all the feature keys requested
    """
    
    # read the features.json as dict
    with open(paths.features_file) as features_file:
        features_dict = json.load(features_file)
        
    # add only included features to a list
    feature_keys = []
    for k in include_keys:
        feature_keys.extend(features_dict[k])
    
    # remove excluded features from the list
    for k in exclude_keys:
        features_to_remove = features_dict[k]
        for f in features_to_remove:
            feature_keys.remove(f)
        
    return feature_keys