

class CFG:
    # save files
    output_dir = "./output"
    log_filename = f"{output_dir}/train.log"

    # console log
    print_freq = 1000

    # seed
    seed = 42

    # transforms
    num_workers = 4 # not use

    # model
    bandpass_params = dict(lf=20, hf=500, order=8, sr=2048)
    qtransform_params = {
        "sr": 2048,
        "fmin": 20,
        "fmax": 1024,
        "hop_length": 16,
        "bins_per_octave": 8,
    }
    model_name = "tf_efficientnet_b0_ns"
    in_chans = 3
    target_size = 1
    target_col = "target"

    # train
    train = True
    n_fold = 4
#    trn_fold = [i for i in range(n_fold)]
    trn_fold = [0]
    epochs = 5
    batch_size = 128

    if model_name == "tf_efficientnet_b0_ns":
        batch_size = 256
        epochs = 10
    
    if model_name == "tf_efficientnet_b4_ns":
        batch_size = 256

    if model_name == "tf_efficientnet_b7_ns":
        if qtransform_params["hop_length"] == 32:
            batch_size = 128
        elif qtransform_params["hop_length"] == 16:
            batch_size = 64
        
    lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-3
    gradient_accumulation_steps = 1
    max_grad_norm = 1000

    # scheduler
    scheduler = "CosineAnnealingLR"
    T_max = 3

    # wandb
    project = 'kaggle-g2net-gravitational-wave-detection'

