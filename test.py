import os.path

from network import init_model
import matplotlib.pyplot as plt
import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset
from monai.networks.blocks import Warp
from dataset import load_train_val_file, get_transforms


print_config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward(batch_data, model):
    warp_layer = Warp().to(device)
    fixed_image = batch_data["fixed_image"].to(device)
    moving_image = batch_data["moving_image"].to(device)
    moving_label = batch_data["moving_label"].to(device)

    # predict DDF through LocalNet
    ddf = model(torch.cat((moving_image, fixed_image), dim=1))

    # warp moving image and label with the predicted ddf
    pred_image = warp_layer(moving_image, ddf)
    pred_label = warp_layer(moving_label, ddf)

    return ddf, pred_image, pred_label


def save_niigz(data, src_file, save_path, mode="pred"):
    import SimpleITK as sitk
    sitkimg = sitk.GetImageFromArray(data)
    nii_info = sitk.ReadImage(src_file)
    # sitkimg.CopyInformation(nii_info)
    sitkimg.SetOrigin(nii_info.GetOrigin())
    sitkimg.SetSpacing(nii_info.GetSpacing())
    sitkimg.SetDirection(nii_info.GetDirection())
    basename = os.path.basename(src_file)
    save_file = f"{save_path}/{basename}"
    if mode == "pred":
        save_file = f"{save_path}/pred_{basename}"
    sitk.WriteImage(sitkimg, save_file)





def infer():
    # dst = "./weights/pair_lung_ct.pth"
    dst = "./train_result_bone/best_metric_model.pth"
    model = init_model()
    model.load_state_dict(torch.load(dst, map_location="cpu"))
    model = model.to(device)

    train_files, valid_files = load_train_val_file()
    train_transforms = get_transforms(mode="train")
    valid_transforms = get_transforms(mode="valid")

    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    # train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=valid_files, transform=valid_transforms, cache_rate=1.0, num_workers=0)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            if i > 2:
                break
            print(i)
            val_ddf, val_pred_image, val_pred_label = forward(val_data, model)
            val_pred_image = val_pred_image.cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_pred_label = val_pred_label.cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_moving_image = val_data["moving_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_moving_label = val_data["moving_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_fixed_image = val_data["fixed_image"].cpu().numpy()[0, 0].transpose((1, 0, 2))
            val_fixed_label = val_data["fixed_label"].cpu().numpy()[0, 0].transpose((1, 0, 2))

            save_path = "./train_result_bone"
            save_niigz(val_fixed_image, val_data["fixed_image_meta_dict"]["filename_or_obj"],save_path)
            save_niigz(val_fixed_label, val_data["fixed_label_meta_dict"]["filename_or_obj"], save_path)

            save_niigz(val_moving_image, val_data["moving_image_meta_dict"]["filename_or_obj"], save_path)
            save_niigz(val_moving_label, val_data["moving_label_meta_dict"]["filename_or_obj"], save_path)

            save_niigz(val_pred_image, val_data["fixed_image_meta_dict"]["filename_or_obj"], save_path, "pred")
            save_niigz(val_pred_label, val_data["fixed_label_meta_dict"]["filename_or_obj"], save_path, "pred")

            print(f"moving_image shape: {val_moving_image.shape}, " f"moving_label shape: {val_moving_label.shape}")
            print(f"fixed_image shape: {val_fixed_image.shape}, " f"fixed_label shape: {val_fixed_label.shape}")
            print(f"val_pred_image shape: {val_pred_image.shape}, " f"val_pred_label shape: {val_pred_label.shape}")

            for depth in range(10):
                print(f"{depth}")
                depth = depth * 10
                # plot the slice [:, :, 80]
                plt.figure("check", (18, 6))
                plt.subplot(1, 6, 1)
                plt.title(f"moving_image {i} d={depth}")
                plt.imshow(val_moving_image[:, :, depth], cmap="gray")
                plt.subplot(1, 6, 2)
                plt.title(f"moving_label {i} d={depth}")
                plt.imshow(val_moving_label[:, :, depth])
                plt.subplot(1, 6, 3)
                plt.title(f"fixed_image {i} d={depth}")
                plt.imshow(val_fixed_image[:, :, depth], cmap="gray")
                plt.subplot(1, 6, 4)
                plt.title(f"fixed_label {i} d={depth}")
                plt.imshow(val_fixed_label[:, :, depth])
                plt.subplot(1, 6, 5)
                plt.title(f"pred_image {i} d={depth}")
                plt.imshow(val_pred_image[:, :, depth], cmap="gray")
                plt.subplot(1, 6, 6)
                plt.title(f"pred_label {i} d={depth}")
                plt.imshow(val_pred_label[:, :, depth])
                plt.savefig('./train_result_bone/subplots_figure.png', dpi=300)
                plt.show()


if __name__ == '__main__':
    infer()
