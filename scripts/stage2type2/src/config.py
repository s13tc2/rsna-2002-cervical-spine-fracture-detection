import os

class cfg:
  gpu_id = int(os.environ["LOCAL_RANK"])

  DEBUG = False
  kernel_type = '0920_2d_lstmv22headv2_convnn_224_15_6ch_8flip_augv2_drl3_rov1p2_rov3p2_bs4_lr6e5_eta6e6_lw151_50ep'
  load_kernel = None
  load_last = True

  fold = 0
  n_folds = 5
  backbone = 'convnext_nano'

  image_size = 224
  n_slice_per_c = 15
  in_chans = 6

  init_lr = 23e-5
  eta_min = 23e-6
  lw = [15, 1]
  batch_size = 8
  drop_rate = 0.
  drop_rate_last = 0.3
  drop_path_rate = 0.
  p_mixup = 0.5
  p_rand_order = 0.2
  p_rand_order_v1 = 0.2

  data_dir = '../data/cropped_2d_224_15_ext0_5ch_0920_2m/cropped_2d_224_15_ext0_5ch_0920_2m/'
  use_amp = True
  num_workers = 4
  out_dim = 1

  n_epochs = 50

  log_dir = '../logs'
  model_dir = '../models'