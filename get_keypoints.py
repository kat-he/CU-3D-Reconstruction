import numpy as np
import torch
from image_processing import *
from overlap_detection_crop import *

def find_common_keypoints(keypoint_image1, keypoint_image2, keypoint_image3):
    keypoint_image1_rounded = keypoint_image1.round().int().cpu()
    keypoint_image2_rounded = keypoint_image2.round().int().cpu()
    keypoint_image3_rounded = keypoint_image3.round().int().cpu()
    keypoints1_np = keypoint_image1_rounded.squeeze(0).numpy()
    keypoints2_np = keypoint_image2_rounded.squeeze(0).numpy()
    keypoints3_np = keypoint_image3_rounded.squeeze(0).numpy()

    common_keypoints = np.intersect1d(
        keypoints1_np.view(np.dtype((np.void, keypoints1_np.dtype.itemsize * keypoints1_np.shape[1]))),
        keypoints2_np.view(np.dtype((np.void, keypoints2_np.dtype.itemsize * keypoints2_np.shape[1]))),
        return_indices=False)
    common_keypoints = np.intersect1d(
        keypoints3_np.view(np.dtype((np.void, keypoints3_np.dtype.itemsize * keypoints3_np.shape[1]))),
        keypoints2_np.view(np.dtype((np.void, keypoints2_np.dtype.itemsize * keypoints2_np.shape[1]))),
        return_indices=False)
    
    common_keypoints = common_keypoints.view(keypoints3_np.dtype).reshape(-1, 2)
    common_keypoints_tensor = torch.from_numpy(common_keypoints).int()

    common_keypoints_tensor = common_keypoints_tensor.unsqueeze(0)
    return common_keypoints_tensor


def pre_processing_keypoints_matches(
        images_list, 
        pairs, 
        extractor, 
        matcher, 
        rotation_detector, 
        pixel_bound=30,
        verbose=False,
        rotation_correction=True,
    ):    
    with h5py.File("keypoints.h5", mode="w") as f_kp, h5py.File("descriptors.h5", mode="w") as f_desc:  
        for image_path in images_list:
            with torch.inference_mode():
                image = load_image(image_path).to(DEVICE)

                # pre-prcessing
                if rotation_correction:
                    image_before_to_after[image_path] = rotate_image(image, feats)
                    image = rotate_image(image, rotation)
                
                # feature enhancement
                brightened_image = increase_brightness(image, value=40)
                contrast_image = increase_contrast(image, factor=2.5)
                hsv_image = rgb_to_hsv_opencv(image)
                clahe_grayscale_image = apply_clahe_grayscale(image)
                clahe_color_image = apply_clahe_color(image)

                # night images
                if image.mean() < pixel_bound:
                    if verbose:
                        print("Image mean before: ", image.mean())
                    image = apply_clahe_color(img)
                    if verbose:
                        print("Image mean after: ", image.mean())
                
                brightened_image_feats = extractor.extract(brightened_image)
                contrast_image_feats = extractor.extract(contrast_image)
                hsv_image_feats = extractor.extract(hsv_image)
                clahe_grayscale_image_feats = extractor.extract(clahe_grayscale_image)
                clahe_color_feats = extractor.extract(clahe_color_image)

                common_keypoints = find_common_keypoints(,
                    find_common_keypoints(
                        brightened_image_feats["keypoints"], 
                        contrast_image_feats["keypoints"], 
                        hsv_image_feats["keypoints"],
                    ), 
                    clahe_grayscale_image_feats["keypoints"],
                    clahe_color_feats["keypoints"],
                )
                
                f_kp[image_path.name] = common_keypoints.reshape(-1, 2).detach().cpu().numpy()
                f_desc[image_path.name] = feats["descriptors"].reshape(len(f_kp[image_path.name]), -1).detach().cpu().numpy()
                

    with h5py.File("keypoints.h5", mode="r") as f_kp, h5py.File("descriptors.h5", mode="r") as f_desc, \
         h5py.File("matches.h5", mode="w") as f_matches:  
        
        # pairs of images
        for pair in pairs:
            key1, key2 = images_list[pair[0]].name, images_list[pair[1]].name
            kp1 = torch.from_numpy(f_kp[key1][...]).to(DEVICE)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(DEVICE)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(DEVICE)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(DEVICE)

            with torch.inference_mode():                
                _, idxs = matcher(desc1, desc2, KF.laf_from_center_scale_ori(kp1[None]), KF.laf_from_center_scale_ori(kp2[None]))

            if len(idxs): group = f_matches.require_group(key1)
            if len(idxs) >= MIN_MATCHES: group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))


