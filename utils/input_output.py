# Imports
import sys
from pathlib import Path
import json
import uproot
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Imports from this project
sys.path.insert(0, Path(__file__).parent.parent)
from utils import paths

def load_features_dict():
    """Load the dictionary of all feature keys (features.json)

    Returns:
        dict
    """
    with open(paths.features_file, "r") as file:
        features_dict = json.load(file)
    return features_dict

def load_feature_keys(include_keys, exclude_keys=None):
    """Read in selected features from the features.json

    Args:
        include_keys (list): Which types of features to include (keys of the json file)
        exclude_keys (list, optional): Which types of features to exclude

    Returns:
        list: A concatenated list of all the feature keys requested
    """
    
    features_dict = load_features_dict()
        
    # add only included features to a list
    feature_keys = []
    for k in include_keys:
        feature_keys.extend(features_dict[k])
    
    # remove excluded features from the list
    if exclude_keys is not None:
        for k in exclude_keys:
            features_to_remove = features_dict[k]
            for f in features_to_remove:
                feature_keys.remove(f)
                
    # remove all duplicates and only keep the first occurence
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    feature_keys = list(dict.fromkeys(feature_keys))
        
    return feature_keys

def load_feature_properties():
    assert paths.feature_properties_file.is_file(), "There is no feature_properties.json. First execute feature_properties.py"
    
    with open(paths.feature_properties_file, "r") as file:
        fprops = json.load(file)
        
    return fprops


def load_data_from_root(file_path, tree_key="DecayTree", features=None, N_entries_max=np.Infinity, batch_size=100000):
    """Read in data from a ROOT Tree as a Pandas Dataframe

    Args:
        file_path (pathlib.Path, str): path to the input root file
        tree_key (str, optional): key of the tree inside the root file, Defaults to "DecayTree".
        features (list(str), optional): list of features to be loaded
        N_entries_max (int, optional): maximal number of entries to be loaded
        batch_size (int, optional): step_size in uproot.iterate, Defaults to 100000.

    Returns:
        pandas.DataFrame
    """
    
    # Check how many entries should be loaded
    with uproot.open(file_path)[tree_key] as tree:
        N_entries_in_tree = tree.num_entries

    N_entries = np.min([N_entries_in_tree, N_entries_max])

    N_batches_estimate = np.ceil(N_entries / batch_size).astype(int)

    print(f"Entries in the data: {N_entries_in_tree}")
    print(f"Entries to be loaded: {N_entries}")

    # Read the input data

    df = pd.DataFrame()
    with uproot.open(file_path, 
                     file_handler=uproot.MultithreadedFileSource, 
                     num_workers=10)[tree_key] as tree:
        tree_iter = tree.iterate(entry_stop=N_entries, 
                                 step_size=batch_size, 
                                 filter_name=features,
                                 library="pd")
        for temp_df in tqdm(tree_iter, "Load Batches", total=N_batches_estimate):
            df = pd.concat([df, temp_df])
    
    return df

def load_preprocessed_data(features=None, N_entries_max=np.Infinity, batch_size=1000000):
    """Read in the already preprocessed data

    Args:
        features (list(str), optional): features to be loaded
        N_entries_max (int, optional): maximal number of entries to be loaded
        batch_size (int, optional): step_size in uproot.iterate, Defaults to 100000.

    Returns:
        pandas.DataFrame
    """
    if isinstance(features, list):
        features = features.copy()
        features.extend(["index", "event_id", "track_id"])
    
    df = load_data_from_root(paths.preprocessed_data_file, features=features, N_entries_max=N_entries_max, batch_size=batch_size)
    df.set_index("index", inplace=True)
    
    return df

