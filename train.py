import os
import torch
from monai.data import DataLoader, Dataset, CacheDataset
from monai.networks.blocks import Warp

from network import init_model
from dataset import get_transforms, load_train_val_file
from loss_optimizer_metric import set_loss, set_metric, set_optimizer, set_lr_scheduler
from configs import cfg

cfg = cfg.train


def load_data():
    train_files, valid_files = load_train_val_file()
    train_transforms = get_transforms(mode="train")
    valid_transforms = get_transforms(mode="valid")
    # 加载数据
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=cfg.num_workers)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    val_ds = CacheDataset(data=valid_files, transform=valid_transforms, cache_rate=1.0, num_workers=0)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    valid_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return train_loader, valid_loader


def set_model_warp():
    model = init_model()
    dst = "./weights/pair_lung_ct.pth"
    model.load_state_dict(torch.load(dst, map_location="cpu"))
    model = model.to(cfg.device)
    warp_layer = Warp().to(cfg.device)
    return model, warp_layer


def forward(batch_data, model, warp_layer):
    fixed_image = batch_data["fixed_image"].to(cfg.device)
    moving_image = batch_data["moving_image"].to(cfg.device)
    moving_label = batch_data["moving_label"].to(cfg.device)

    # predict DDF through LocalNet
    ddf = model(torch.cat((moving_image, fixed_image), dim=1))

    # warp moving image and label with the predicted ddf
    pred_image = warp_layer(moving_image, ddf)
    pred_label = warp_layer(moving_label, ddf)

    return ddf, pred_image, pred_label


def train_one_epoch(model, warp_layer, train_loader, optimizer, image_loss, label_loss, regularization):
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()

        ddf, pred_image, pred_label = forward(batch_data, model, warp_layer)
        pred_label[pred_label > 1] = 1

        fixed_image = batch_data["fixed_image"].to(cfg.device)
        fixed_label = batch_data["fixed_label"].to(cfg.device)
        fixed_label[fixed_label > 0] = 1
        loss = (
                image_loss(pred_image, fixed_image) + 100 * label_loss(pred_label,
                                                                       fixed_label) + 10 * regularization(ddf)
        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    return epoch_loss


def valid_one_epoch(model, warp_layer, valid_loader, dice_metric):
    model.eval()
    with torch.no_grad():
        for val_data in valid_loader:
            val_ddf, val_pred_image, val_pred_label = forward(val_data, model, warp_layer)
            val_pred_label[val_pred_label > 1] = 1

            val_fixed_image = val_data["fixed_image"].to(cfg.device)
            val_fixed_label = val_data["fixed_label"].to(cfg.device)
            val_fixed_label[val_fixed_label > 0] = 1
            dice_metric(y_pred=val_pred_label, y=val_fixed_label)

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        return metric


def run():
    # ########################
    # load dataset
    train_loader, valid_loader = load_data()  # 读取数据
    # set model
    model, warp_layer = set_model_warp()  # 设置模型
    # set loss、 optimizer、metric
    image_loss, label_loss, regularization = set_loss()
    optimizer = set_optimizer(cfg.optimizer, model, cfg.lr)
    scheduler = set_lr_scheduler(cfg.scheduler, optimizer, cfg.max_epochs, cfg.gamma)
    dice_metric = set_metric()

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(cfg.max_epochs):
        if (epoch + 1) % cfg.valid_interval == 0 or epoch == 0:
            metric = valid_one_epoch(model, warp_layer, valid_loader, dice_metric)
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(cfg.save_path, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} "
                f"current mean dice: {metric:.4f}\n"
                f"best mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

        print("-" * 10)
        print(f"epoch {epoch + 1}/{cfg.max_epochs}")
        train_loss = train_one_epoch(model, warp_layer, train_loader, optimizer, image_loss, label_loss, regularization)
        scheduler.step()
        epoch_loss_values.append(train_loss)
        print(f"epoch {epoch + 1} average loss: {train_loss:.4f}")


if __name__ == '__main__':
    run()
