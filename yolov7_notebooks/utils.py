import matplotlib.pyplot as plt
import json
import xmltodict
from pathlib import Path
import os, os.path

def to_yolov7_bbox(bbox, w, h):
    """
    retuns a bounding box but centeralized wrt image size
    :param bbox: bbox is a list containing all bounding box info
        is a lost of [xmin, ymin, xmax, ymax]
    :param w: width of image
    :param h: height of image
    :return: normalized bounding box list
    """
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def create_sample_xml(doc):
    """
    Is called by create_metadata and creates an image_name.xml which stores annotations info:
        - Class label
        - bounding box
    The filename is the corrosponsing image

    Create a sample dictionary from the given xml file
    :param doc: A dictionary containing all sample information
    :return: a string
    """
    doc = doc['annotation']
    categories_names = {'D00': 0, 'D10': 1, 'D40': 2, 'D20': 3, 'pothole': 4}  # map between category name and int
    sample = list()
    if "object" not in doc:
        return []  # return empty list

    filename = doc["filename"][:-4]  # remove .jpg
    width = int(doc["size"]["width"])
    height = int(doc["size"]["height"])

    objs = doc["object"]
    if isinstance(objs, dict):
        objs = [objs]

    for obj in objs:

        category_id = categories_names[obj["name"]]
        bbox = [int(float(obj["bndbox"]["xmin"])),
                int(float(obj["bndbox"]["ymin"])),
                int(float(obj["bndbox"]["xmax"])),
                int(float(obj["bndbox"]["ymax"]))]
        bbox_v7 = to_yolov7_bbox(bbox, width, height)
        bbox_v7_str = " ".join([str(x) for x in bbox_v7])
        sample.append(f"{category_id} {bbox_v7_str}")


    return sample


def create_metadata(path_labels, path_yolo_folder):
    """
    :param path_imgs: full path to the images
    :param path_labels: full path to the labels
    """
    for file in Path(path_labels).glob('*'):
        with open(file) as fd:
            doc = xmltodict.parse(fd.read(), process_namespaces=True)
            sample = create_sample_xml(doc)

            if sample:
                filename = file.name[:-4] + ".txt"
                path_file = Path(path_yolo_folder) / filename
                with open(path_file, "w") as f:
                    f.write("\n".join(sample))

def get_dic(txt_path, mode='preds', img_path='../datasets/Norway/test/images'):
    file_dic = {}

    for file_name in os.listdir(img_path):
        new_name = file_name[:-4] + '.txt'
        file_dic[new_name] = []
        path = txt_path + '/' + new_name
        if not os.path.exists(path):
            with open(path, 'w') as fp:
                fp.write(" ")
        else:
            file = open(path, 'r')
            lines = file.readlines()
            for line in lines:
                line_arr = line.split(' ')
                if len(line_arr) > 3:
                    label = int(line_arr[0])
                    a, b, c, d = line_arr[1:5]
                    if mode=='preds':
                        conf = float(line_arr[-1][:-1])
                        file_dic[new_name].append([label, a, b, c, d, conf])
                    else:
                        file_dic[new_name].append([label, a, b, c, d])
                else: # has no labels
                    # print(new_name, "NO PRED")
                    pass
    return file_dic
