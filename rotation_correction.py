rot_correction_transform = T.Compose([
    T.Resize((224, 224)),
    T.ConvertImageDtype(torch.float),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def perform_rotation_correction(img):
    """
    param: img - Tensor output from load_torch_image()
    return: rotated image
    """
    model = create_model("swsl_resnext50_32x4d")
    model.eval();
    
    # apply transformation
    img = rot_correction_transform(img)
    
    # predict rotation
    with torch.no_grad():
        prediction = model(img).numpy()
    predicted_rot_deg = prediction[0].argmax() * 90
    
    #     rot_matrix = [
    #         [np.cos(-predicted_rot_deg*np.pi/180), -np.sin(-predicted_rot_deg*np.pi/180), 0],
    #         [np.sin(-predicted_rot_deg*np.pi/180), np.cos(-predicted_rot_deg*np.pi/180), 0],
    #         [0, 0, 1],
    #     ]
    #     # 0:0
    #     # 90:
    #     # 180:
    #     # 270:
    #     # 360: -pi/2 = 0
    
    # apply rotation: T.functional.rotate - default is counterclockwise
    if predicted_rot_deg == 90:
        img = T.functional.rotate(img, 360-90)
    elif predicted_rot_deg == 180:
        img = T.functional.rotate(img, 360-180)
    elif predicted_rot_deg == 270:
        img = T.functional.rotate(img, 360-270)
    return img, predicted_rot_deg


def _testing_rotation_correction():
    import pandas as pd
    from utils import load_torch_image
    train_dir = "/kaggle/input/image-matching-challenge-2024/train"
    ground_truth = pd.read_csv(f"{train_dir}/train_labels.csv")
    ground_truth.index = ground_truth['image_name']

    img = load_torch_image(f"{train_image_dir}/{file_name}")
    rotated_img, predicted_rot_angle = perform_rotation_correction(img)
    print(predicted_rot_angle)
    media.show_image(rotated_img[0, ...].permute(1,2,0).cpu())
