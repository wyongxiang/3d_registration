import numpy as np
import os

from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
)

from configs import cfg
cfg = cfg.datasets


def get_transforms(mode=""):
    if mode == "train":
        train_transforms = Compose(
            [
                LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                           ensure_channel_first=True,
                           image_only=True),
                ScaleIntensityRanged(
                    keys=["fixed_image", "moving_image"],
                    a_min=-700,
                    a_max=1300,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                RandAffined(
                    keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                    mode=("bilinear", "bilinear", "nearest", "nearest"),
                    prob=1.0,
                    # spatial_size=(192, 192, 208), # x, y, z
                    spatial_size=(384, 384, 384),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1),
                ),
                Resized(
                    keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                    mode=("trilinear", "trilinear", "nearest", "nearest"),
                    align_corners=(True, True, None, None),
                    # spatial_size=(96, 96, 104),
                    spatial_size=(128, 128, 128),
                ),
            ]
        )
        return train_transforms
    elif mode == "valid" or "test":
        val_transforms = Compose(
            [
                LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                           ensure_channel_first=True,
                           image_only=True),
                ScaleIntensityRanged(
                    keys=["fixed_image", "moving_image"],
                    a_min=-700,
                    a_max=1300,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resized(
                    keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                    mode=("trilinear", "trilinear", "nearest", "nearest"),
                    align_corners=(True, True, None, None),
                    # spatial_size=(96, 96, 104),
                    spatial_size=(128, 128, 128),
                ),
            ]
        )
        return val_transforms
    else:
        print("set mode train、valid、test")
        pass


def load_train_val_file():
    data_dir = cfg.data_path
    '''
    data_dicts = [
        {
            "fixed_image": os.path.join(data_dir, "scans/case_%03d_exp.nii.gz" % idx),
            "moving_image": os.path.join(data_dir, "scans/case_%03d_insp.nii.gz" % idx),
            "fixed_label": os.path.join(data_dir, "lungMasks/case_%03d_exp.nii.gz" % idx),
            "moving_label": os.path.join(data_dir, "lungMasks/case_%03d_insp.nii.gz" % idx),
        }
        for idx in range(1, 21)
    ]

    train_files, valid_files = data_dicts[:18], data_dicts[18:]
    '''
    data_dicts = [
        {
            "fixed_image": os.path.join(data_dir, "images_AD/case_%03d_A.nii.gz" % idx),
            "moving_image": os.path.join(data_dir, "images_AD/case_%03d_D.nii.gz" % idx),
            "fixed_label": os.path.join(data_dir, "masks_AD/case_%03d_A.nii.gz" % idx),
            "moving_label": os.path.join(data_dir, "masks_AD/case_%03d_D.nii.gz" % idx),
        }
        for idx in range(1, 12)
    ]

    train_files, valid_files = data_dicts[:8], data_dicts[8:]
    return train_files, valid_files


def test():
    import matplotlib.pyplot as plt
    from monai.data import DataLoader, Dataset, CacheDataset
    from monai.utils import set_determinism, first

    train_files, valid_files = load_train_val_file()
    train_transforms = get_transforms(mode="train")
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    fixed_image = check_data["fixed_image"][0][0].permute(1, 0, 2)
    fixed_label = check_data["fixed_label"][0][0].permute(1, 0, 2)
    moving_image = check_data["moving_image"][0][0].permute(1, 0, 2)
    moving_label = check_data["moving_label"][0][0].permute(1, 0, 2)

    print(f"moving_image shape: {moving_image.shape}, " f"moving_label shape: {moving_label.shape}")
    print(f"fixed_image shape: {fixed_image.shape}, " f"fixed_label shape: {fixed_label.shape}")

    # plot the slice [:, :, 50]
    plt.figure("check", (12, 6))
    plt.subplot(1, 4, 1)
    plt.title("moving_image")
    plt.imshow(moving_image[:, :, 50], cmap="gray")
    plt.subplot(1, 4, 2)
    plt.title("moving_label")
    plt.imshow(moving_label[:, :, 50])
    plt.subplot(1, 4, 3)
    plt.title("fixed_image")
    plt.imshow(fixed_image[:, :, 50], cmap="gray")
    plt.subplot(1, 4, 4)
    plt.title("fixed_label")
    plt.imshow(fixed_label[:, :, 50])

    plt.show()
    plt.show()


if __name__ == '__main__':
    test()
