
import os
import sys

import cv2
import glob
import json
import numpy as np

from tqdm import tqdm

import pycocotools.mask as mutils


def get_items(pred, img_id, id):
    items = []
    mask = pred.round().astype(np.uint8)
    nc, label = cv2.connectedComponents(mask, connectivity=8)
    for c in range(nc):
        if np.all(mask[label == c] == 0):
            continue
        else:
            ann = np.asfortranarray((label == c).astype(np.uint8))
            rle = mutils.encode(ann)
            bbox = [int(_) for _ in mutils.toBbox(rle)]
            area = int(mutils.area(rle))
            score = float(pred[label == c].mean())
            items.append({
                "segmentation": {
                    "size": [int(_) for _ in rle["size"]],
                    "counts": rle["counts"].decode()},
                "bbox": [int(_) for _ in bbox], "area": int(area), "iscrowd": 0, "category_id": 1,
                "image_id": int(img_id), "id": id+len(items),
                "score": float(score)
            })
    return items


def main(res_root):
    files = glob.glob(os.path.join(res_root, '*.png'))
    all_res = []
    for file in tqdm(files, total=len(files)):
        img_id = int(os.path.basename(file).split('.')[0])
        pred = cv2.imread(file, -1)
        pred[pred>0] = 1
        items = get_items(pred, img_id, len(all_res))
        all_res.extend(items)
    return all_res


if __name__ == "__main__":
    res_root = sys.argv[1]
    results_root = os.path.join(res_root, 'results/')
    if not os.path.exists(results_root):
        os.mkdir(results_root)

    all_res = main(res_root)
    coco_file = os.path.join(results_root, "test.segm.json")
    with open(coco_file, "w") as f:
        json.dump(all_res, f)

    os.system(f"cd {res_root} && zip -9 -r results.zip results/")
