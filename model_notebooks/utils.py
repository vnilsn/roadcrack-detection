import matplotlib.pyplot as plt
import json
import xmltodict
from pathlib import Path

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556],
          [0.466, 0.674, 0.188]]


def create_sample(doc):
    """
    Is called by create_metadata in COCO format

    Create a sample dictionary from the given xml file
    :param doc: A dictionary containing all sample information
    :return: a dictionary connecting images with objects and bounding boxes
    """
    doc = doc['annotation']
    categories_names = {'D00': 0, 'D10': 1, 'D40': 2, 'D20': 3, 'pothole': 4}  # map between category name and int

    if "object" not in doc:
        return {}  # return empty dict

    image_id = int(doc['filename'][-10:-4])
    sample = dict()
    sample["file_name"] = doc["filename"]
    objects = dict()
    categories = list()

    objs = doc["object"]
    if isinstance(objs, dict):
        objs = [objs]

    annos = 1
    sample["annotations"] = list()
    for obj in objs:
        annotation = {}
        annotation["id"] = image_id + annos
        annos += 1
        annotation["image_id"] = image_id
        annotation["category_id"] = categories_names[obj["name"]]
        annotation["bbox"] = [int(float(obj["bndbox"]["xmin"])),
                              int(float(obj["bndbox"]["ymin"])),
                              int(float(obj["bndbox"]["xmax"])) - int(float(obj["bndbox"]["xmin"])),
                              int(float(obj["bndbox"]["ymax"])) - int(float(obj["bndbox"]["ymin"]))]

        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        annotation["iscrowd"] = 0
        sample["annotations"].append(annotation)

    return sample


def create_metadata(path_imgs, path_labels):
    """
    Writes a metadata file for the given images and labels
    :param path_imgs: full path to the images
    :param path_labels: full path to the labels
    """
    samples = list()
    for file in Path(path_labels).glob('*'):
        with open(file) as fd:
            doc = xmltodict.parse(fd.read(), process_namespaces=True)
            sample = create_sample(doc)
            if sample:
                samples.append(sample)
            # move file to directory '../unlabeled'
            else:
                # If doc['annotation']['filename'] in unlabeld directory, do nothing
                if not Path(path_imgs).joinpath('../unlabeled', doc['annotation']['filename']).exists():
                    #Path(file).rename(Path(path_imgs).joinpath('../unlabeled', doc['annotation']['filename']))
                    name = path_imgs[:-7] + 'unlabeled/' + doc['annotation']['filename']
                    Path(path_imgs + doc["annotation"]["filename"]).rename(name)

    filename = path_imgs + "metadata.jsonl"
    with open(filename, 'w') as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote metadata.jsonl to {path_imgs}")


def get_datalader(path_images, path_labels=None, batch_size=32):
    """
    Read images and labels from the given
    paths and return a dataloader
    :param path_images: filepath to images
    :param path_labels: filepath to labels, if any
    :param batch_size:
    :return: dataloader object ready for training / testing
    """
    pass


def plot_results(image, results):
    """
    Plot image with bounding boxes and labels
    :param image: single image
    :param results: output of the pipeline object for the model
    """
    # plt.figure(figsize=(16,10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100
    for result, color in zip(results, colors):
        box = result['box']
        xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
        label = result['label']
        prob = result['score']
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=3))
        text = f'{label}: {prob:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
