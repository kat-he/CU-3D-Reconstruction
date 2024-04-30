from pathlib import Path
from copy import deepcopy
import numpy as np
import math
import pandas as pd
import pandas.api.types
from itertools import combinations
import sys, torch, h5py, pycolmap, datetime
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia as K
import kornia.feature as KF
from lightglue.utils import load_image
from lightglue import LightGlue, ALIKED, match_pair
from transformers import AutoImageProcessor, AutoModel
from check_orientation.pre_trained_models import create_model
from database import *
from h5_to_db import *
from image_processing import *
from get_keypoints import *
from get_image_pairs import *
from ransac_and_sparse_reconstruction import *
from rotation_correction import *
from utils import *
from maa_metrics_utils import*

# SIMILLIAR PAIRS
EXHAUSTIVE = True
MIN_PAIRS = 50
DISTANCES_THRESHOLD = 0.3
TOLERANCE = 500

# KEYPOINTS EXTRACTOR AND MATCHER
MAX_NUM_KEYPOINTS = 4096
RESIZE_TO = 1280
DETECTION_THRESHOLD = 0.005
MIN_MATCHES = 100
MIN_MATCHES_OVERLAP = 5000

# RANSAC AND SPARSE RECONSTRUCTION
MIN_MODEL_SIZE = 5
MAX_NUM_MODELS = 3

# CROSS VALIDATION
N_SAMPLES = 15


IMC_PATH = '/kaggle/input/image-matching-challenge-2024'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    data_path,
    get_pairs,
    keypoints_matches,
    ransac_and_sparse_reconstruction,
):
    results = {}
    data_dict = parse_sample_submission(data_path)
    datasets = list(data_dict.keys())

    extractor = ALIKED(max_num_keypoints=MAX_NUM_KEYPOINTS,detection_threshold=DETECTION_THRESHOLD,resize=RESIZE_TO).eval().to(DEVICE)
    matcher = KF.LightGlueMatcher("aliked", {'width_confidence':-1, 'depth_confidence':-1, 'mp':True if 'cuda' in str(DEVICE) else False}).eval().to(DEVICE)
    rotation_detector = create_model("swsl_resnext50_32x4d").eval().to(DEVICE)
    
    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}
            
        for scene in data_dict[dataset]:
            images_dir = data_dict[dataset][scene][0].parent
            results[dataset][scene] = {}

            # get images
            image_paths = data_dict[dataset][scene]
            index_pairs = get_pairs(image_paths)

            # pre-processing pipeline + keypoint matching + overlap detection
            pre_processing_keypoints_matches(images_list, 
                index_pairs, 
                extractor, 
                matcher, 
                rotation_detector, 
                pixel_bound=30,
                verbose=False,
                rotation_correction=True,
            )  
            
            # 3D reconstruction               
            maps = ransac_and_sparse_reconstruction(image_paths[0].parent)
            clear_output(wait=False)
            
            path = 'test' if submit else 'train'
            images_registered  = 0
            best_idx = 0
            for idx, rec in maps.items():
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx
                    
            for k, im in maps[best_idx].images.items():
                key = Path(IMC_PATH) / path / scene / "images" / im.name
                results[dataset][scene][key] = {}
                results[dataset][scene][key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
                results[dataset][scene][key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

            create_submission(results, data_dict, Path(IMC_PATH))


if __name__ == '__main__':
    def main()