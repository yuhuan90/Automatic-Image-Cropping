
import os
import torch.utils.data as data
import cv2
import math
import numpy as np

MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (123, 117, 104)


class TransformFunction(object):

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

        scale = 256.0 / min(image.shape[:2])
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image,(int(w),int(h)))
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean

        scale_height = float(resized_image.shape[0]) / image.shape[0]
        scale_width = float(resized_image.shape[1]) / image.shape[1]

        transformed_bbox = {}
        transformed_bbox['xmin'] = []
        transformed_bbox['ymin'] = []
        transformed_bbox['xmax'] = []
        transformed_bbox['ymax'] = []
        MOS = []
        for annotation in annotations:
            annotation_split = annotation.split()
            transformed_bbox['xmin'].append(math.floor(float(annotation_split[1]) * scale_width))
            transformed_bbox['ymin'].append(math.floor(float(annotation_split[0]) * scale_height))
            transformed_bbox['xmax'].append(math.ceil(float(annotation_split[3]) * scale_width))
            transformed_bbox['ymax'].append(math.ceil(float(annotation_split[2]) * scale_height))

            MOS.append((float(annotation_split[-1]) - MOS_MEAN) / MOS_STD)

        resized_image = resized_image.transpose((2, 0, 1))
        return {'image': resized_image, 'bbox': transformed_bbox, 'MOS': MOS}



class TransformFunctionTest(object):

    def __call__(self, image, bboxes):

        scale = 256.0 / min(image.shape[:2])
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image,(int(w),int(h)))
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean

        scale_height = float(resized_image.shape[0]) / image.shape[0]
        scale_width = float(resized_image.shape[1]) / image.shape[1]

        transformed_bboxes = {}
        transformed_bboxes['xmin'] = []
        transformed_bboxes['ymin'] = []
        transformed_bboxes['xmax'] = []
        transformed_bboxes['ymax'] = []

        for bbox in bboxes:
            transformed_bboxes['xmin'].append(math.floor(float(bbox[1]) * scale_width))
            transformed_bboxes['ymin'].append(math.floor(float(bbox[0]) * scale_height))
            transformed_bboxes['xmax'].append(math.ceil(float(bbox[3]) * scale_width))
            transformed_bboxes['ymax'].append(math.ceil(float(bbox[2]) * scale_height))

        resized_image = resized_image.transpose((2, 0, 1))
        return resized_image,transformed_bboxes


def generate_bboxes(image):

    bins = 12.0
    h = image.shape[0]
    w = image.shape[1]
    step_h = h / bins
    step_w = w / bins
    annotations = list()
    for x1 in range(0,4):
        for y1 in range(0,4):
            for x2 in range(8,12):
                for y2 in range(8,12):
                    if (x2-x1)*(y2-y1)>0.4999*bins*bins and (y2-y1)*step_w/(x2-x1)/step_h>0.5 and (y2-y1)*step_w/(x2-x1)/step_h<2.0:
                        annotations.append([round(step_h*(0.5+x1)),round(step_w*(0.5+y1)),round(step_h*(0.5+x2)),round(step_w*(0.5+y2))])

    return annotations

class setup_test_dataset(data.Dataset):

    def __init__(self, dataset_dir='testsetDir',
                 transform=TransformFunctionTest(), augmentation = None):
        self.dataset_dir = dataset_dir
        image_lists = os.listdir(self.dataset_dir)
        self._imgpath = list()
        self._annopath = list()
        for image in image_lists:
          self._imgpath.append(os.path.join(self.dataset_dir, image))
        self.transform = transform
        self.augmentation = augmentation


    def __getitem__(self, idx):
        image = cv2.imread(self._imgpath[idx])

        if self.augmentation:
            image = self.augmentation(image)

        bboxes = generate_bboxes(image)

        # to rgb
        image = image[:, :, (2, 1, 0)]


        if self.transform:
            resized_image,transformed_bboxes = self.transform(image,bboxes)

        sample = {'imgpath': self._imgpath[idx], 'image': image, 'resized_image': resized_image, 'bboxes': bboxes, 'tbboxes': transformed_bboxes}

        return sample

    def __len__(self):
        return len(self._imgpath)

