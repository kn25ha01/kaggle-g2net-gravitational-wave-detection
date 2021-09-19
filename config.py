

class CFG:
    # debug
    debug = False

    print_freq = 1000
    num_workers = 4
    seed = 42

    # model
    model_name = "tf_efficientnet_b0_ns"
    target_size = 1
    qtransform_params = {"sr": 2048, "fmin": 20, "fmax": 1024, "hop_length": 32, "bins_per_octave": 8}

    # dataset
    target_col = "target"

    # train
    train = True
    n_fold = 4
    trn_fold = [0, 1, 2, 3]
    epochs = 1
    batch_size = 256
    lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-3
    gradient_accumulation_steps = 1
    max_grad_norm = 1000

    # scheduler
    scheduler = "CosineAnnealingLR"
    T_max = 3
