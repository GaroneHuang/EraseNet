import os
from tqdm import tqdm
import numpy as np
import cv2


data_type = "SCUT-EnsText"
path = "SCUT-EnsText/train"


assert data_type=="SCUT-EnsText" or data_type=="SCUT-Syn"


if data_type=="SCUT-EnsText":
    os.makedirs(os.path.join(path, "mask"), exist_ok=True)
    file_names = list(map(lambda x: x.split(".")[0], os.listdir(os.path.join(path, "all_images"))))
    for file_name in tqdm(file_names):
        f_gt = open(os.path.join(path, "all_gts", file_name+".txt"), "r", encoding="utf-8")
        boxes = list(map(lambda x: np.array(list(map(int, x.strip().split(",")))).reshape(-1, 2), f_gt.readlines()))
        f_gt.close()
        img = cv2.imread(os.path.join(path, "all_images", file_name+".jpg"))
        mask = np.ones(img.shape)
        cv2.fillPoly(mask, boxes, (0., 0., 0.))
        cv2.imwrite(os.path.join(path, "mask", file_name+".jpg"), (mask*255.).astype(np.uint8))