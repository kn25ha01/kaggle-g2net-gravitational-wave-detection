

class CFG:
    # save files
    output_dir = "./output"
    log_filename = f"{output_dir}/train.log"

    # console log
    print_freq = 100

    # seed
    seed = 42

    # transforms
    num_workers = 4 # not use

    # model
    qtransform_params = {
        "sr": 2048,
        "fmin": 20,
        "fmax": 1024,
        "hop_length": 32,
        "bins_per_octave": 8,
    }
    model_name = "tf_efficientnet_b0_ns"
    in_chans = 3
    target_size = 1
    target_col = "target"

    # train
    train = True
    n_fold = 4
    trn_fold = [0] # [i for i in range(n_fold)]
    epochs = 1
    batch_size = 128
    if model_name == "tf_efficientnet_b0_ns": batch_size = 1024
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

