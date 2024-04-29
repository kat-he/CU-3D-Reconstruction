def ransac_and_sparse_reconstruction(images_path):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    db_name = f'colmap_{time_str}.db'
    db = COLMAPDatabase.connect(db_name)
    db.create_tables()
    fname_to_id = add_keypoints(db, '/kaggle/working/', images_path, '', 'simple-pinhole', False)
    add_matches(db, '/kaggle/working/',fname_to_id)
    db.commit()
    
    pycolmap.match_exhaustive(db_name, sift_options={'num_threads':1})
    maps = pycolmap.incremental_mapping(
        database_path=db_name, 
        image_path=images_path,
        output_path='/kaggle/working/', 
        options=pycolmap.IncrementalPipelineOptions({'min_model_size':MIN_MODEL_SIZE, 'max_num_models':MAX_NUM_MODELS, 'num_threads':1})
    )
    return maps