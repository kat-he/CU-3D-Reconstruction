def get_pairs(images_list, is_exhaustive, device=DEVICE):
    if is_exhaustive:
        return get_all_pairs(images_list, device=DEVICE)
    return get_best_pairs(images_list, device=DEVICE)


def get_all_pairs(images_list, device=DEVICE):
    return list(combinations(range(len(images_list)), 2)) 
    

def get_best_pairs(images_list, device=DEVICE):
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1/')
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1/').eval().to(DEVICE)
    embeddings = []
    
    for img_path in images_list:
        image = K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=DEVICE)[None, ...]
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False ,do_resize=True, 
                               do_center_crop=True, size=224).to(DEVICE)
            outputs = model(**inputs)
            embedding = F.normalize(outputs.last_hidden_state.max(dim=1)[0])
        embeddings.append(embedding)
        
    embeddings = torch.cat(embeddings, dim=0)
    distances = torch.cdist(embeddings,embeddings).cpu()
    distances_ = (distances <= DISTANCES_THRESHOLD).numpy()
    np.fill_diagonal(distances_,False)
    z = distances_.sum(axis=1)
    idxs0 = np.where(z == 0)[0]
    for idx0 in idxs0:
        t = np.argsort(distances[idx0])[1:MIN_PAIRS]
        distances_[idx0,t] = True
        
    s = np.where(distances >= TOLERANCE)
    distances_[s] = False
    
    idxs = []
    for i in range(len(images_list)):
        for j in range(len(images_list)):
            if distances_[i][j]:
                idxs += [(i,j)] if i<j else [(j,i)]
    
    idxs = list(set(idxs))
    return idxs