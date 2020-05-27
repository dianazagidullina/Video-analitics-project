from flask import *
from flask_wtf import FlaskForm
from wtforms import SelectField
from wtforms import StringField
from wtforms import IntegerField
from flask_wtf import FlaskForm
import cv2
import os
import numpy
import cv2
import sys
import numpy as np
from mrcnn.config import Config
from mrcnn.visualize import display_instances
import json
import webcolors
import colorsys
import mrcnn.model as modellib
import random
from matplotlib import pyplot as plt
from matplotlib import patches
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog



ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

class FashionConfig(Config):
    NUM_CATS = 46
    IMAGE_SIZE = 512

    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1  # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4  # a memory error occurs when IMAGES_PER_GPU is too high

    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # DETECTION_NMS_THRESHOLD = 0.0

    # STEPS_PER_EPOCH should be the number of instances
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200


class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


IMAGE_SIZE = 512


def makeModel():
    global model

    ROOT_DIR = os.path.abspath("../")
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)
    model_path = "model/mask_rcnn_fashion_0007.h5"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

 #Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#
# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")
#
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

with open("label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret string'


# app.cache.clear()

# GLOBAL
filename = ""
model = None
clothes = set()
images_to_show = []
res_time = set()

class MyForm(FlaskForm):

    amount = SelectField('Select amount', choices=[
        ("1", 1),
        ("2", 2),
        ("3", 3),
        ("4", 4),
        ("5", 5),
        ("6", 6),
        ("7", 7),
        ("8", 8),
        ("9", 9),
        ("10", 10)
    ])

    category = SelectField('Select category', choices=[
        ("1", "shirt, blouse"),
        ("2", "top, t-shirt, sweatshirt"),
        ("3", "sweater"),
        ("4", "cardigan"),
        ("5", "jacket"),
        ("6", "vest"),
        ("7", "pants"),
        ("8", "shorts"),
        ("9", "skirt"),
        ("10", "coat"),
        ("11", "dress"),
        ("12", "jumpsuit"),
        ("13", "glasses"),
        ("14", "hat"),
        ("15", "glove"),
        ("16", 'belt'),
        ("17", 'tights, stockings'),
        ("18", 'bag, wallet'),
        ("19", 'shoe'),
        ("20", 'scarf')
    ])

    colour = SelectField('Select colour', choices=[
        ("1", "aqua"),
        ("2", "black"),
        ("3", "blue"),
        ("4", "fuchsia"),
        ("5", "green"),
        ("6", "gray"),
        ("7", "lime"),
        ("8", "maroon"),
        ("9", "navy"),
        ("10", "olive"),
        ("11", "purple"),
        ("12", "red"),
        ("13", "silver"),
        ("14", "teal"),
        ("15", "white"),
        ("16", 'yellow'),
        ("17", 'orange')
    ])

@app.route('/')
def start():

    return redirect(url_for('form'))

@app.route('/form', methods=['GET', 'POST'])
def form():
    form = MyForm(request.form)
    global filename, images_to_show, res_time
    if request.method == 'POST' and 'file' in request.files:
        f = request.files['file']
        f.save(f.filename)
        filename = f.filename

        amount = int(form.amount.data)

        categories = request.form.getlist('category')
        colours = request.form.getlist('colour')
        for i in range(amount):
            category = dict(form.category.choices).get(categories[i])
            colour = dict(form.colour.choices).get(colours[i])
            clothes.add((category, colour))

        processVideo(filename)

        if not res_time:
            return render_template('show_images.html', image_name=images_to_show, found="No")
        else:
            res_time_s = str(res_time)
            res_time_s = res_time_s[1:-1]
            return render_template('show_images.html', image_name=images_to_show, found="Yes", res=res_time_s)

    return render_template("form.html", form=form)


class Cloth:
    def __init__(self, category, colour):
        self.category = category
        self.colour = colour

def get_colour_name(rgb_triplet):
    min_colours = {}
    for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def make_image_smaller(img, scale_percent=60):
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  # resize image
  resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  return resized



def resize_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img

def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

def getCategories(r):
    cloth_ids = r['class_ids'] - 1
    names = numpy.array(label_names)[cloth_ids.astype(int)]
    return names

def getClothes(img, masks, categories):
    result = set()
    for k in range(masks.shape[2]):
        colours = {}
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                if masks[i][j][k]:
                    r, g, b = (img[i, j])
                    colour = get_colour_name((r, g, b))
                    if colour in colours:
                        colours[colour] += 1
                    else:
                        colours[colour] = 1
        if len(colours) != 0:
            result.add((categories[k], max(colours, key=lambda key: colours[key])))
    return result

def processClothes(image_names):
    global clothes
    found = False
    for i in range(len(image_names)):
        image_name = image_names[i]
        image_path = "frames/" + image_name
        img = cv2.imread(image_path)
        width = img.shape[1]
        height = img.shape[0]
        if min(width, height > 100):
            img = make_image_smaller(img, 100 * 100 / min(width, height))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = model.detect([resize_image(image_path)])
        r = result[0]
        if r['masks'].size > 0:
            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'),
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            y_scale = img.shape[0] / IMAGE_SIZE
            x_scale = img.shape[1] / IMAGE_SIZE
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

            masks, rois = refine_masks(masks, rois)
        else:
            masks, rois = r['masks'], r['rois']

        display_instances("frames/" + image_name, img, rois, masks, r['class_ids'],
                          ['bg'] + label_names, r['scores'],
                          figsize=(12, 12))

        categories = getCategories(r)
        img_cloths = getClothes(img, masks, categories)

        print(str(i) + " person: " +  str(img_cloths))
        if (clothes.issubset(img_cloths)):
            print(str(i) + " OK")
            found = True
        else:
            print(str(i) + " not OK")

        print(" ")
    return found


def findPeople(image_name):
    global images_to_show
    im = cv2.imread("frames/" + image_name)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    instances = outputs["instances"]
    N = len(instances)
    boxes_with_people = []
    indices = []
    for i in range(N):
        # 0 class is person
        if instances.pred_classes[i] == 0 and instances.scores[i] > 0.95:
            boxes_with_people.append(instances.pred_boxes[i].tensor.tolist()[0])
            indices.append(i)
    print("boxes: " + str(boxes_with_people))
    person_instances = instances[indices]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(person_instances.to("cpu"))
    person_amount = len(boxes_with_people)
    image_names = []

    for person in range(person_amount):
        boxe = boxes_with_people[person]
        crop_img = im[int(boxe[1]):int(boxe[3]), int(boxe[0]):int(boxe[2])].copy()
        name = image_name[:-4] + '-' + str(person) + '.jpg'
        cv2.imwrite("frames/" + name, crop_img)
        image_names.append(name)

    res = processClothes(image_names)
    image_names.insert(0, image_name)
    images_to_show.append(image_names)
    return res


def processImage(image_name):
    print("processing " + image_name)
    return findPeople(image_name)


def processVideo(filename):
    makeModel()
    print("video processing..")
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    seconds = 1
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
    multiplier = int(fps * seconds)
    # print("multiplier", multiplier)
    #################### Initiate Process ################
    image_names = []
    while success:
        frameId = int(round(vidcap.get(1)))  # current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        success, image = vidcap.read()
        if frameId % multiplier == 0:
            id = str(int(frameId) // int(multiplier))
            print("Saved " + id + " second of the video")
            image_names.append("%s.jpg" % id)
            cv2.imwrite(r"frames/%s.jpg" % id, image)

    vidcap.release()


    for i in range(len(image_names)):
        if processImage(image_names[i]):
            # print("Found")
            res_time.add(i * seconds + 1)



@app.route('/process/<filename>')
def send_image(filename):
    return send_from_directory("frames", filename, cache_timeout=0)


if __name__ == '__main__':
    app.run(debug = True)
