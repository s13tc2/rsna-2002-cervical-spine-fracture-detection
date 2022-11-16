from config import cfg
import albumentations

def get_train_transforms():
  return albumentations.Compose([
      albumentations.Resize(cfg.image_size, cfg.image_size),
      albumentations.HorizontalFlip(p=0.5),
      albumentations.VerticalFlip(p=0.5),
      albumentations.Transpose(p=0.5),
      albumentations.RandomBrightness(limit=0.1, p=0.7),
      albumentations.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),

      albumentations.OneOf([
          albumentations.MotionBlur(blur_limit=3),
          albumentations.MedianBlur(blur_limit=3),
          albumentations.GaussianBlur(blur_limit=3),
          albumentations.GaussNoise(var_limit=(3.0, 9.0)),
      ], p=0.5),
      albumentations.OneOf([
          albumentations.OpticalDistortion(distort_limit=1.),
          albumentations.GridDistortion(num_steps=5, distort_limit=1.),
      ], p=0.5),

      albumentations.Cutout(max_h_size=int(cfg.image_size * 0.5), max_w_size=int(cfg.image_size * 0.5), num_holes=1, p=0.5),
  ])

def get_valid_transforms():
  return albumentations.Compose([
      albumentations.Resize(cfg.image_size, cfg.image_size),
  ])