import argparse
from pathlib import Path
import numpy as np
import os
import cv2
import torch
import easyocr

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, increment_path, scale_coords
from utils.torch_utils import select_device


def detect(model, imgsz, root_path, sub_path, mask_save_path):
    source = os.path.join(root_path, sub_path)
    mask_save_dir = Path(increment_path(Path(mask_save_path) / sub_path))
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset:
        print("Processing:" + path)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            mask = np.zeros(im0.shape)
            p = Path(p)  # to Path
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    mask[c1[1]:c2[1], c1[0]:c2[0]] = 1
                
                ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1] * mask.shape[2])
                print(ratio)
                if ratio > opt.mask_ratio_thres: 
                    print(str(p)+" contains a large part of human. Remove this frame.")
                    if os.path.isfile(p):
                        os.remove(p)
                    break

            print("No large human in: " + str(p))

            # OCR mask
            result = reader.readtext(str(p))
            if len(result) != 0:
                for det in result:
                    coordinates = det[0]
                    c1, c2 = (int(coordinates[0][0]), int(coordinates[0][1])), (int(coordinates[2][0]), int(coordinates[2][1]))
                    mask[c1[1]:c2[1], c1[0]:c2[0], :] = 1

            name = os.path.splitext(os.path.basename(p))[0].split('_')
            mask_save_path = os.path.join(mask_save_dir, name[0]+'_mask_'+name[2]+ '.jpg')
            cv2.imwrite(mask_save_path, mask * 255)


if __name__ == '__main__':
    # python detect.py --conf-thres 0.80 
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--root_path', type=str, default='image/', help='source')
    parser.add_argument('--mask_save_path', type=str, default='mask', help='source')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.60, help='IOU threshold for NMS')
    parser.add_argument('--mask_ratio_thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # class == 'person'
    parser.add_argument('--classes', default=0, nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')

    opt = parser.parse_args()
    
    # Load model
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    reader = easyocr.Reader(['en'])

    if half:
        model.half()  # to FP16

    folders_dir = sorted(os.listdir(opt.root_path))
    for i, sub_dir in enumerate(folders_dir):
        try:
            with torch.no_grad():
                detect(model, imgsz, opt.root_path, sub_dir, opt.mask_save_path)
            print('Finished Video: ' + str(i))
        except:
            print('Failed Video: ' + str(i))

# python human_ocr_mask.py    