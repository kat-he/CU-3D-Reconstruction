def increase_brightness(image, value=40):
    if image.dtype != torch.float32:
        image = image.float() / 255.0
    adjustment = (value / 255.0)
    brightened_image = image + adjustment
    brightened_image = torch.clamp(brightened_image, 0, 1)
    return brightened_image


def increase_contrast(image, factor=2.5):
    mean = torch.mean(image, dim=[2, 3], keepdim=True)
    contrasted_image = (image - mean) * factor + mean
    contrasted_image = torch.clamp(contrasted_image, 0, 1)
    return contrasted_image


def rgb_to_hsv_opencv(image_tensor):
    is_cuda = image_tensor.is_cuda
    if is_cuda:
        image_tensor = image_tensor.cpu()
    image_np = image_tensor.permute(0, 2, 3, 1).numpy()  # Change shape to (N, H, W, 3) for OpenCV
    image_np = (image_np * 255).astype(np.uint8)
    hsv_images = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in image_np]
    hsv_tensor = torch.from_numpy(np.array(hsv_images)).float() / 255.0  # Normalize back to [0, 1]
    hsv_tensor = hsv_tensor.permute(0, 3, 1, 2)  # Change shape back to (N, 3, H, W) for PyTorch
    
    if is_cuda:
        hsv_tensor = hsv_tensor.to('cuda')
        
    return hsv_tensor


def apply_clahe_grayscale(image):
    img = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    return clahe_img


def apply_clahe_color(image):
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img) # split the LAB image to different channels
    
    # apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # merge the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img = cv2.merge((cl, a, b))
    
    # convert LAB image back to RGB
    clahe_img_rgb = cv2.cvtColor(updated_lab_img, cv2.COLOR_LAB2BGR)
    return clahe_img_rgb