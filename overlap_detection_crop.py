def overlap_detection(extractor, matcher, image_0, image_1, min_matches):
    feats_0, feats_1, matches_01 = match_pair(extractor, matcher, image_0, image_1)
    
    if len(matches_01['matches']) < min_matches:
        return feats_0, feats_1, matches_01
    
    # phase 1: initial match
    keypoints_0, keypoints_1, matches = feats0["keypoints"], feats1["keypoints"], matches_01["matches"]
    matched_kpts_0, matched_kpts_1 = keypoints_0[matches[..., 0]], keypoints_1[matches[..., 1]]
    
    # get bounds for crop
    left_0, top_0 = matched_kpts_0.numpy().min(axis=0).astype(int)
    width_0, height_0 = matched_kpts_0.numpy().max(axis=0).astype(int)
    height_0 -= top_0
    width_0 -= left_0
    left_1, top_1 = m_kpts_1.numpy().min(axis=0).astype(int)
    width_1, height_1 = m_kpts_1.numpy().max(axis=0).astype(int)
    height_1 -= top_1
    width_1 -= left_1

    # crop
    crop_box_0 = (top_0, left_0, height_0, width_0)
    crop_box_1 = (top_1, left_1, height_1, width_1)
    cropped_img_tensor_0 = TF.crop(image0, *crop_box_0)
    cropped_img_tensor_1 = TF.crop(image1, *crop_box_1)

    # phase 2: match on cropped region
    feats0_c, feats1_c, matches01_c = match_pair(
        extractor, matcher, cropped_img_tensor_0, cropped_img_tensor_1)
    feats0_c['keypoints'][...,0] += left_0
    feats0_c['keypoints'][...,1] += top_0
    feats1_c['keypoints'][...,0] += left_1
    feats1_c['keypoints'][...,1] += top_1
    return feats0_c, feats1_c, matches01_c