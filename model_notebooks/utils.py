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
    doc = doc['annotations']
    categories_names = {'D00': 0, 'D10': 1, 'D40': 2, 'D20': 3, 'pothole': 4}  # map between category name and int

    if "object" not in doc:
        return {}  # return empty dict


    sample = dict()
    sample["image"] = doc["filename"]
    image_id = int(doc['filename'][-10:-4])
    sample["image_id"] = image_id

    objs = doc["object"]
    if isinstance(objs, dict):
        objs = [objs]

    annos = 1
    sample["annotations"] = list()
    for obj in objs:
        annotation = {}
        annotation["id"] = str(image_id + annos)
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
    TODO: Make it work for test data

    Writes a metadata file for the given images and labels
    :param path_imgs: full path to the images
    :param path_labels: full path to the labels
    """
    samples = list()
    for file in Path(path_labels).glob('*'):
        with open(file) as fd:
            doc = xmltodict.parse(fd.read(), process_namespaces=True)
            sample = create_sample(doc)
            sample["image"] = str(Path(path_imgs) / sample["image"])
            if sample:
                samples.append(sample)

    filename = "metadata.jsonl"
    with open(filename, 'w') as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote metadata.jsonl with {len(samples)} samples to {Path.cwd()}")


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
