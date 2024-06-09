import os
import shutil
from tqdm import tqdm
import numpy as np


def get_A_D_V_data(root_dir, save_path):
    save_image_path = f"{save_path}/images"
    save_mask_path = f"{save_path}/masks"
    if not os.path.exists(save_path):
        os.makedirs(save_image_path, exist_ok=True)
        os.makedirs(save_mask_path, exist_ok=True)
    patients_id = os.listdir(root_dir)
    pbar = tqdm(total=len(patients_id))
    for idx, patient_id in enumerate(patients_id):
        patient_id_path = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_id_path):
            continue
        image_file = f"{patient_id_path}/image_preprocessed/image.nii.gz"
        mask_file = f"{patient_id_path}/bone_mask/mask.nii.gz"
        if not os.path.exists(image_file) or not os.path.exists(mask_file):
            continue
        new_patient_id = patient_id.split(patient_id[0])[-1]
        new_patient_id = new_patient_id.split("_")[0]
        new_image_file = f"{save_image_path}/{new_patient_id}_{patient_id[0]}.nii.gz"
        new_mask_file = f"{save_mask_path}/{new_patient_id}_{patient_id[0]}.nii.gz"
        shutil.copy(image_file, new_image_file)
        shutil.copy(mask_file, new_mask_file)

        pbar.update(1)
    pbar.close()

    return


def run():
    root = r"/mnt/NAS/OperationPlanGroup/AI训练及相关数据/腹部训练-汤静/肾集合系/2020411_宋晓斌下载的积水结石狭窄输尿管数据"

    sub_dir_list = [r"/标注用数据1/_已标注数据-宋晓斌/",
                    r"/标注用数据2/_已标注数据-宋晓斌/",
                    r"/标注用数据2/_已标注数据-张敏/",
                    ]
    save_path = r"/fdata/wyx/data/registration_bone_data"
    for sub_dir in sub_dir_list:
        root_dir = f"{root}/{sub_dir}"
        get_A_D_V_data(root_dir, save_path)


if __name__ == '__main__':
    run()
