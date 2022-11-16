import os


class cfg:
    gpu_id = int(os.environ["LOCAL_RANK"])

    DEBUG = False

    kernel_type = "0920_1bonev2_effv2s_224_15_6ch_augv2_mixupp5_drl3_rov1p2_bs8_lr23e5_eta23e6_50ep"
    load_kernel = None
    load_last = True

    fold = 0
    n_folds = 5
    backbone = "tf_efficientnetv2_s_in21ft1k"

    image_size = 224
    n_slice_per_c = 15
    in_chans = 6

    init_lr = 23e-5
    eta_min = 23e-6
    batch_size = 8
    drop_rate = 0.0
    drop_rate_last = 0.3
    drop_path_rate = 0.0
    p_mixup = 0.5
    p_rand_order_v1 = 0.2

    data_dir = (
        "../data/cropped_2d_224_15_ext0_5ch_0920_2m/cropped_2d_224_15_ext0_5ch_0920_2m"
    )
    use_amp = True
    num_workers = 4
    out_dim = 1

    n_epochs = 75

    log_dir = "../logs"
    model_dir = "../models"
