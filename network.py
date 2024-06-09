from monai.networks.nets import LocalNet


def init_model(mode="LocalNet"):
    if mode == "LocalNet":
        net = LocalNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=32,
            extract_levels=(3,),
            out_activation=None,
            out_kernel_initializer="zeros",
        )
    else:
        net = None

    return net


if __name__ == '__main__':
    model = init_model(mode="LocalNet")
    print(f"model:{model}")
