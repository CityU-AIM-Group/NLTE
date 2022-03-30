import os
from collections import defaultdict

import numpy as np
from PIL import Image
import shutil
import xml.etree.ElementTree as ET


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

class VOCDataset:
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root, split='trainval', base_dir='.', keep_difficult=True, img_ext='.jpg', dataset_name=''):
        self.root = root
        self.split = split
        self.keep_difficult = keep_difficult
        self.img_ext = img_ext

        # voc_root = os.path.join(self.root, base_dir)
        voc_root = self.root
        images_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')
        # super().__init__(images_dir, dataset_name)

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, split + '.txt')
        with open(os.path.join(split_f), "r") as f:
            ids = [x.strip() for x in f.readlines()]
        self.ids = ids

        cat_ids = list(range(len(VOCDataset.CLASSES)))
        self.label2cat = {
            label: cat for label, cat in enumerate(cat_ids)
        }

    def get_annotations_by_image_id(self, img_id):
        ann_path = os.path.join(self.annotation_dir, img_id + '.xml')
        target = self.parse_voc_xml(ET.parse(ann_path).getroot())['annotation']
        img_info = {
            'width': target['size']['width'],
            'height': target['size']['height'],
            'id': img_id,
            'file_name': img_id + self.img_ext,
        }

        boxes = []
        labels = []
        difficult = []
        for obj in target['object']:
            is_difficult = bool(int(obj['difficult']))
            if is_difficult and not self.keep_difficult:
                continue
            label_name = obj['name']
            if label_name not in self.CLASSES:
                continue
            difficult.append(is_difficult)
            label_id = self.CLASSES.index(label_name)
            box = obj['bndbox']
            box = list(map(lambda x: float(x) - 1, [box['xmin'], box['ymin'], box['xmax'], box['ymax']]))
            boxes.append(box)
            labels.append(label_id)
        boxes = np.array(boxes).reshape((-1, 4))
        labels = np.array(labels)
        difficult = np.array(difficult)

        return {'img_info': img_info, 'boxes': boxes, 'labels': labels, 'difficult': difficult}

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag: {k: v if k == 'object' else v[0] for k, v in def_dic.items()}
            }
        elif node.text:
            text = node.text.strip()
            voc_dict[node.tag] = text
        return voc_dict

def count_instances(img_level_annos, total_instances): # Add all instances to a list
    for instance_index_in_img in range(img_level_annos['labels'].shape[0]):
        instance = {}
        instance["file_name"] = img_level_annos['img_info']['file_name'][:-3] + 'xml' # change .jpg to .xml
        instance["boxes"] = img_level_annos["boxes"][instance_index_in_img]
        instance["labels"] = img_level_annos["labels"][instance_index_in_img]
        # print(instance)
        total_instances.append(instance)
    
import random
from copy import deepcopy
seed = 1997
random.seed(seed)

def generate_sym_noise(VOC_dir, sym_noise_percent):
    total_instances = []
    voc2007 = VOCDataset(VOC_dir)
    classes_idxs = list(np.arange(len(voc2007.CLASSES)))
    # DEFAULTS
    for img_id in voc2007.ids:
        img_level_annos = voc2007.get_annotations_by_image_id(img_id)
        count_instances(img_level_annos, total_instances)
    # Sample noisy samples
    noise_instances = random.sample(total_instances, int(len(total_instances) * sym_noise_percent))
    # get original anno files
    anno_dir = os.path.join(VOC_dir, 'Annotations')
    anno_list = os.listdir(anno_dir)
    # make noise dir
    noisy_anno_dir = os.path.join(VOC_dir, 'Noisy_Annotations_per_{:3f}'.format(sym_noise_percent))
    print('making noisy dir {}'.format(noisy_anno_dir))
    if not os.path.exists(noisy_anno_dir):
        os.mkdir(noisy_anno_dir)
        # one-by-one copy
        for clean_anno in anno_list:
            shutil.copy(os.path.join(anno_dir, clean_anno), os.path.join(noisy_anno_dir, clean_anno))
    # change the idx in obj
    for instance in noise_instances:
        tree = ET.parse(os.path.join(noisy_anno_dir, instance['file_name']))
        root = tree.getroot()
        for obj in root.findall('object'):
            box = obj.findall("bndbox")[0]
            xmin = int(box[0].text) - 1
            ymin = int(box[1].text) - 1
            xmax = int(box[2].text) - 1
            ymax = int(box[3].text) - 1
            box = np.array([xmin, ymin, xmax, ymax])
            if (box == instance["boxes"]).all(): # This obj is the instance that we want to make it noisy
                classes_idxs_new = deepcopy(list(classes_idxs))
                classes_idxs_new.remove(instance["labels"])
                desired_noise = int(random.sample(classes_idxs_new, 1)[0])
                if desired_noise != 0:
                    obj[0].text = voc2007.CLASSES[desired_noise]
                else:
                    root.remove(obj)
        tree.write(os.path.join(noisy_anno_dir, instance['file_name']), 'UTF-8')
    print('Done generating for {} in dataset {}!'.format(sym_noise_percent, VOC_dir))

if __name__ == "__main__":
    generate_sym_noise('./VOC2007', 0.2)
    generate_sym_noise('./VOC2007', 0.4)
    generate_sym_noise('./VOC2007', 0.6)
    generate_sym_noise('./VOC2007', 0.8)
    generate_sym_noise('./VOC2012', 0.2)
    generate_sym_noise('./VOC2012', 0.4)
    generate_sym_noise('./VOC2012', 0.6)
    generate_sym_noise('./VOC2012', 0.8)
    