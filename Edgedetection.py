import copy
import math
import os

import cv2
import numpy as np
import tensorflow as tf
import random
import Gates
import morphology
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from termcolor import colored
from time import time
def start():
    detection_model, category_index = loadModel()
    #startall(detection_model, category_index)
    startsolo(detection_model, category_index)

def startall(detection_model, category_index):
    for i in os.listdir("images"):
        start_time = time()
        startsolo(detection_model, category_index,str(i))
        end_time = time()
        spent_time = round(end_time - start_time,2)
        print("Es wurden " + str(spent_time) +" Sekunden für die Verarbeitung des Bildes benötigt")
        print("\n")

def startsolo(detection_model, category_index,IMAGE_name = "24c5c7e5-54e5-43ad-bcb0-a89e6e8714c5"):
    global IMAGE_NAME
    global IMAGE_PATH
    global MINSCORE
    global def_dir_offset
    MINSCORE = 0.75
    def_dir_offset = 6
    IMAGE_NAME = IMAGE_name
    IMAGE_PATH = os.path.join("images", IMAGE_NAME)
    print("Image: " + colored(IMAGE_NAME,"blue") + " with Min-Detections-Score of: " + str(MINSCORE))
    #detections, image_np = load(IMAGE_NAME, IMAGE_PATH)
    image_np = load(IMAGE_NAME, IMAGE_PATH)
    detections = run_inference(detection_model, image_np)
    detections = detectFromImage(category_index, detection_model, image_np, detections)
    boxes, imageblack = drawboxes(detections, image_np)
    pre = preprocessing(imageblack)
    skel = drawedges(pre, boxes)
    classified_pixels, port_pixels = getclassifiedpixels(skel)
    trivial_sections, crossing_pixels_in_port_sections, = edge_sections_identify(
        classified_pixels, port_pixels)
    crossing_pixels_in_port_sections, boundingboxes = merge_crossingpixels(crossing_pixels_in_port_sections,
                                                                           classified_pixels)
    merged_sections = traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, port_pixels,
                                         boundingboxes)
    edge_sections = trivial_sections + merged_sections
    classified_pixelsa = copy.deepcopy(classified_pixels)
    for a in edge_sections:
        for b in a:
            classified_pixelsa[b[0], b[1]] = 10
        classified_pixelsa = copy.deepcopy(classified_pixels)


    connections = connectNodes(edge_sections, boxes)
    drawconections(edge_sections,image_np)
    elements = build(boxes, connections)
    solve(elements, boxes, image_np)
    TabelleColored(elements)

def drawconections(edge_sections,image_np):
    Colors =[]
    counterc = 0
    img_con = image_np.copy()
    r = lambda: random.randint(0, 4) * 64
    while len(Colors) < len(edge_sections):
        curColor =(r(), r(), r())
        if sum(list(curColor)) >= 641:
            continue
        if curColor in Colors:
            continue
        Colors.append(curColor)
    for edge_section in edge_sections:
        for px in edge_section:
            img_con[px[0], px[1]] = Colors[counterc]
            img_con[px[0], px[1] + 1] = Colors[counterc]
            img_con[px[0] + 1, px[1]] = Colors[counterc]
            img_con[px[0], px[1] - 1] = Colors[counterc]
            img_con[px[0] - 1, px[1]] = Colors[counterc]
        counterc+=1
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_connections' + '.png'), img_con)


def Tabelle(elements):
    inputelements = []
    outputelements = []

    for element in elements:
        if type(element[0]) == Gates.Input:
            element[0].setFalse()
            inputelements.append(element)
        elif type(element[0]) == Gates.Output:
            outputelements.append(element)
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")
    print("|", end="")
    for element in inputelements:
        print("\tIn: " + str(element[1]), end="\t|")
    print("\t|", end="")
    for element in outputelements:
        print("\tOut:" + str(element[1]), end="\t|")
    print("")
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")
    print("|", end="")
    for element in inputelements:
        print("\t" + str(element[0].getValue()), end="\t|")
    print("\t|", end="")
    for element in outputelements:
        print("\t" + str(element[0].getValue()), end="\t|")
    print("")
    for _ in range(1, int(math.pow(2, len(inputelements)))):
        for _ in range(0, len(inputelements)):
            print("+ - - - - - ", end="")
        print(end="+ - ")
        for _ in range(0, len(outputelements)):
            print("+ - - - - - ", end="")
        print("+")
        scounter = 0
        switched = True
        while switched and scounter < len(inputelements):
            switched = not inputelements[scounter][0].switch()
            scounter += 1
        print("|", end="")
        for element in inputelements:
            print("\t" + str(element[0].getValue()), end="\t|")

        print("\t|", end="")
        for element in outputelements:
            print("\t" + str(element[0].getValue()), end="\t|")
        print("")
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")


def TabelleColored(elements):
    inputelements = []
    outputelements = []

    for element in elements:
        if type(element[0]) == Gates.Input:
            element[0].setFalse()
            inputelements.append(element)
        elif type(element[0]) == Gates.Output:
            outputelements.append(element)
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")
    print("|", end="")
    for element in inputelements:
        print("\tIn: " + str(element[1]), end="\t|")
    print("\t|", end="")
    for element in outputelements:
        print("\tOut:" + str(element[1]), end="\t|")
    print("")
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")
    print("|", end="")
    for element in inputelements:
        print("\t", end="")
        if element[0].getValue() == True:
            print(colored(str(element[0].getValue()), "green"), end="\t|")
        else:
            print(colored(str(element[0].getValue()), "red"), end="\t|")
    print("\t|", end="")
    for element in outputelements:
        print("\t", end="")
        if element[0].getValue() == True:
            print(colored(str(element[0].getValue()), "green"), end="\t|")
        else:
            print(colored(str(element[0].getValue()), "red"), end="\t|")
    print("")
    for _ in range(1, int(math.pow(2, len(inputelements)))):
        for _ in range(0, len(inputelements)):
            print("+ - - - - - ", end="")
        print(end="+ - ")
        for _ in range(0, len(outputelements)):
            print("+ - - - - - ", end="")
        print("+")
        scounter = 0
        switched = True
        while switched and scounter < len(inputelements):
            switched = not inputelements[scounter][0].switch()
            scounter += 1
        print("|", end="")
        for element in inputelements:
            print("\t", end="")
            if element[0].getValue() == True:
                print(colored(str(element[0].getValue()), "green"), end="\t|")
            else:
                print(colored(str(element[0].getValue()), "red"), end="\t|")

        print("\t|", end="")
        for element in outputelements:
            print("\t", end="")
            if element[0].getValue() == True:
                print(colored(str(element[0].getValue()), "green"), end="\t|")
            else:
                print(colored(str(element[0].getValue()), "red"), end="\t|")
        print("")
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")




def TabelleLatex(elements):
    inputelements = []
    outputelements = []

    for element in elements:
        if type(element[0]) == Gates.Input:
            element[0].setFalse()
            inputelements.append(element)
        elif type(element[0]) == Gates.Output:
            outputelements.append(element)

    for element in inputelements:
        print("In: " + str(element[1]), end="&")
    for element in outputelements:
        print("Out:" + str(element[1]), end="&")
    print("")
    for element in inputelements:
        print(str(element[0].getValue()), end="&")
    for element in outputelements:
        print("" + str(element[0].getValue()), end="&")
    print("\hline")
    for _ in range(1, int(math.pow(2, len(inputelements)))):
        print("\hline")
        scounter = 0
        switched = True
        while switched and scounter < len(inputelements):
            switched = not inputelements[scounter][0].switch()
            scounter += 1
        for element in inputelements:
            print(str(element[0].getValue()), end="&")

        for element in outputelements:
            print(str(element[0].getValue()), end="&")
        print("")
    print("\hline")



def loadModel():
    # Patches
    utils_ops.tf = tf.compat.v1
    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    #Import labels and model
    PATH_TO_LABELS = 'SSDMobileNet/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    detection_model = tf.saved_model.load('SSDMobileNet/saved_model')

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore('SSDMobileNet/checkpoint/ckpt-0').expect_partial()
    return detection_model,category_index


def run_inference(model, image):
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # The input needs to be a tensor, convert it using tf.convert_to_tensor.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with tf.newaxis.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run detection
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                 for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    return output_dict



def detectFromImage(category_index, detection_model, image_np, detections):
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],# + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=15,
        min_score_thresh=MINSCORE,
        agnostic_mode=False)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_detections' + '.png'), image_np_with_detections)
    return detections



def connectNodes(edge_sections, boxes):
    connections = {}
    counter = 0
    for section in edge_sections:
        if section[0][1] < section[-1][1]:
            connections[counter] = [[section[0],section[-1]], [-1, -1]]
            # np.array([section[-1][0], section[-1][1]])
            c = 0
            for box in boxes:
                if (box[3] == section[0][1] or box[3] + 1 == section[0][1]) and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][0] = c
                if (box[1] == section[-1][1] + 1 or box[1] == section[-1][1] + 2) and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][1] = c
                c += 1
        else:
            connections[counter] = [[section[-1], section[0]], [-1, -1]]
            #np.array([section[-1][0], section[-1][1]])
            c = 0
            for box in boxes:
                if (box[3] == section[-1][1] or box[3] + 1 == section[-1][1]) and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][0] = c
                if (box[1] == section[0][1] + 1 or box[1] == section[0][1] + 2) and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][1] = c

                c += 1
        counter += 1
    return connections


def build(boxes, connections):
    elements = []
    alt = 0
    label_offset = 1
    for box in boxes:
        if box[4] == 0 + label_offset:
            # print("switch_True")
            elements.append([Gates.Input(True), alt])
        elif box[4] == 1 + label_offset:
            # print("switch_False")
            elements.append([Gates.Input(), alt])
        elif box[4] == 2 + label_offset:
            # print("and")
            elements.append([Gates.And(), alt])
        elif box[4] == 3 + label_offset:
            # print("or")
            elements.append([Gates.Or(), alt])
        elif box[4] == 4 + label_offset:
            # print("nand")
            elements.append([Gates.Nand(), alt])
            # elements[-1].addInput(alt)
        elif box[4] == 5 + label_offset:
            # print("nor")
            elements.append([Gates.Nor(), alt])
        elif box[4] == 6 + label_offset:
            # print("not")
            elements.append([Gates.Not(), alt])
        elif box[4] == 7 + label_offset:
            # print("bulb")
            elements.append([Gates.Output(), alt])
        alt += 1
    for connection in connections.values():
        print("Connect: " + str(connection[1][0]) + " mit " + str(connection[1][1]))
        if not type(elements[connection[1][1]][0]) == Gates.Input:
            elements[connection[1][1]][0].addInput(elements[connection[1][0]][0])
    return elements


def solve(elements, boxes, image_np):
    for element in elements:
        if type(element[0]) == Gates.Input:
            element[0].update()
    print("Lösung")
    count = 0
    for element in elements:
        if type(element[0]) == Gates.Output:
            print("Output " + str(element[1]) + ": " + str(element[0].getValue()))
            cv2.putText(image_np,str(element[0].getValue()),
                (int((boxes[count][1] + boxes[count][3]) / 2 - 20), int((boxes[count][0] + boxes[count][2]) / 2 - 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        count += 1
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_Solution' + '.png'), image_np)


def load(IMAGE_NAME, IMAGE_PATH):
    # with open("images/" + IMAGE_NAME + "/" + IMAGE_NAME + "_detections.pickle", "rb") as file:
    #     detections = pickle.load(file)
    img = cv2.imread(IMAGE_PATH + "/" + IMAGE_NAME + ".jpg")
    image_np = np.array(img)

    #return detections, image_np
    return image_np

def drawboxes(detections, image_np):
    count = 0
    for i in detections['detection_scores']:
        if i >= MINSCORE:
            count += 1
    boxes = np.zeros((count, 5))
    box_pixel = np.zeros((image_np.shape[0], image_np.shape[1], image_np.shape[2]))
    imageblack = image_np.copy()
    image_np2 = image_np.copy()
    for q in range(0, count):
        xmin = detections['detection_boxes'][q][0] * image_np.shape[0] - 3
        ymin = detections['detection_boxes'][q][1] * image_np.shape[1] - 3
        xmax = detections['detection_boxes'][q][2] * image_np.shape[0] + 3
        ymax = detections['detection_boxes'][q][3] * image_np.shape[1] + 3
        cv2.putText(
            image_np2,
            str(q),
            (int((ymin + ymax) / 2), int((xmin + xmax) / 2 - 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3)
        boxes[q] = [int(xmin), int(ymin), int(xmax), int(ymax), detections['detection_classes'][q]]
        #imageblack[int(xmin):int(xmax), int(ymin):int(ymax)] = (0, 0, 0)
        imageblack[int(xmin):int(xmax), int(ymin):int(ymax)] = (255, 255, 255)
        box_pixel[int(xmin):int(xmax), int(ymin):int(ymax)] = (255, 255, 255)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_imageblack' + '.png'), imageblack)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_box_pixel' + '.png'), box_pixel)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_numbers' + '.png'), image_np2)
    return boxes, imageblack

def preprocessing(imageblack):
    imgray = cv2.cvtColor(imageblack, cv2.COLOR_BGR2GRAY)
    otsu_threshold, image_result = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img2 = np.zeros((imageblack.shape[0], imageblack.shape[1], imageblack.shape[2]))
    img2[:, :, 0] = image_result
    img2[:, :, 1] = image_result
    img2[:, :, 2] = image_result
    img2 = 255 - img2
    img2 = img2.astype(np.uint64)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_preprocess' + '.png'), img2)
    return img2


def drawedges(img2, boxes):
    Kanten = np.copy(img2)
    # for q in boxes:
    #     Kanten[int(q[0]):int(q[2]), int(q[1]):int(q[3])] = (0, 0, 0)
    A = np.copy(Kanten) / 255
    A = morphology.zhang_and_suen_binary_thinning(A)
    for x in range(1, A.shape[0] - 1):
        for y in range(1, A.shape[1] - 1):
            if A[x, y, 0] == 1:
                d_1 = (x > 2) and (A[x - 2, y - 1, 0] == 0 or A[x - 2, y, 0] == 1 or A[x - 1, y - 1, 0] == 1)
                d_2 = (y > 2) and (A[x + 1, y - 2, 0] == 0 or A[x, y - 2, 0] == 1 or A[x + 1, y - 1, 0] == 1)
                d_3 = (y < A.shape[1] - 2) and (
                        A[x - 1, y + 2, 0] == 0 or A[x, y + 2, 0] == 1 or A[x - 1, y + 1, 0] == 1)
                d_4 = (y < A.shape[1] - 2) and (
                        A[x + 1, y + 2, 0] == 0 or A[x, y + 2, 0] == 1 or A[x + 1, y + 1, 0] == 1)
                if A[x - 1, y + 1, 0] == 1 and (A[x - 1, y, 0] == 1 and d_1):
                    A[x - 1, y] = [0, 0, 0]
                if A[x - 1, y - 1, 0] == 1 and (A[x, y - 1, 0] == 1 and d_2):
                    A[x, y - 1] = [0, 0, 0]
                if A[x + 1, y + 1, 0] == 1 and (A[x, y + 1, 0] == 1 and d_3):
                    # A[x + 1, y, 0] = A[x, y + 1, 0] = 0
                    # A[x + 1, y, 1] = A[x, y + 1, 1] = 0
                    # A[x + 1, y, 2] = A[x, y + 1, 2] = 0
                    A[x, y + 1] = [0, 0, 0]
                if A[x - 1, y + 1, 0] == 1 and (A[x, y + 1, 0] == 1 and d_4):
                    A[x, y + 1] = [0, 0, 0]

    skel = A
    skel = skel * 255
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_skel' + '.png'), skel)
    return skel


def getclassifiedpixels(skel):
    skel = skel / 255
    height = skel.shape[0]
    width = skel.shape[1]

    classified_pixels = np.zeros((height, width))
    #Todo port_pixels zu NP Array
    port_pixels = []

    for x in range(1, height - 1):
        for y in range(1, width - 1):

            if skel[x, y, 0] == 1:
                # eight neighborhood of pixel (x, y)
                skel_neighborhood = np.array(
                    [skel[x - 1, y, 0], skel[x + 1, y, 0], skel[x, y - 1, 0], skel[x, y + 1, 0], skel[x + 1, y + 1, 0],
                     skel[x + 1, y - 1, 0], skel[x - 1, y + 1, 0], skel[x - 1, y - 1, 0]])
                n0 = np.sum(skel_neighborhood)
                if n0 < 2:
                    classified_pixels[x, y] = 4  # port pixels
                    port_pixels.append((x, y))
                elif n0 == 2:
                    classified_pixels[x, y] = 2  # edge pixels
                elif n0 > 2:
                    classified_pixels[x, y] = 3  # crossing pixels
    return classified_pixels, port_pixels


def edge_sections_identify(classified_pixels, port_pixels):
    trivial_sections = []
    start_pixels = dict.fromkeys(port_pixels, 0)
    crossing_pixels_in_port_sections = {}

    for start in start_pixels:
        # if port pixel is already visited, then continue
        if start_pixels[start] == 1:
            continue
        else:

            start_pixels[start] = 1
            delta = np.array([0, 0])
            section = []
            section.append(start)
            x, y = start
            neighbor = None
            neighbor_value = -float('inf')

            for i in range(0, 3):
                for j in range(0, 3):
                    if (i != 1 or j != 1) and (classified_pixels[x + i - 1, y + j - 1] > neighbor_value):
                        neighbor = np.array([x + i - 1, y + j - 1])
                        neighbor_value = classified_pixels[x + i - 1, y + j - 1]

            next = neighbor
            next_value = neighbor_value
            delta = np.subtract(next, start)

            while next_value == 2:  # edge pixel

                section.append(next)
                neighbor = None
                neighbor_value = -float('inf')
                x, y = next
                for i in range(0, 3):
                    for j in range(0, 3):
                        if (i != 1 or j != 1) and (i != 1 - delta[0] or j != 1 - delta[1]) and (
                                classified_pixels[x + i - 1, y + j - 1] > neighbor_value):
                            neighbor = np.array([x + i - 1, y + j - 1])
                            neighbor_value = classified_pixels[x + i - 1, y + j - 1]
                next_value = neighbor_value
                delta = np.subtract(neighbor, next)
                next = neighbor

            section.append(next)
            last_element = next
            next = last_element
        next_value = classified_pixels[next[0], next[1]]
        if next_value == 4:  # port pixel

            # marks the next pixel as already visited
            start_pixels[(next[0], next[1])] = 1
            trivial_sections.append(section)

        elif next_value == 3:  # crossing pixel
            pos = (next[0], next[1])
            if not pos in crossing_pixels_in_port_sections:
                crossing_pixels_in_port_sections[pos] = []
            #TODO Remove 0 in crossing_pixels_in_port_sections
            crossing_pixels_in_port_sections[pos].append(section)
    start_pixels.clear()

    return trivial_sections, crossing_pixels_in_port_sections


def calcangle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(max(-1,min(dot_product,1)))
    return np.degrees(angle)


def find_missing2(crossing_pixel, sections, classified_pixels, boundingbox):
    for s in sections:
        s[0] = np.array(s[0])
    #TODO FIX FIRST ELEMET OF SECTIONS
    # look around bondingbox form earlier
    #radiust = 3  # smaler circle = 3
    for i in range(boundingbox[0], boundingbox[0] + boundingbox[2]):
        for j in range(boundingbox[1], boundingbox[1] + boundingbox[3]):
            # if ( radiust > i and i > -radiust) and (radiust > j and j > -radiust):
            #     continue
            currentpos = ( i, j)
            if classified_pixels[currentpos] == 3:
                back = currentpos
                radius = 1
                for p in range(-radius, +radius + 1):
                    for q in range(-radius, +radius + 1):
                        if p == 0 and q == 0:
                            continue
                        nextpos = (currentpos[0] + p, currentpos[1] + q)
                        if classified_pixels[nextpos] == 2 and not any(
                                [np.any(np.all(np.array(nextpos) == s, axis=1)) for s in sections]):
                            #TODO CHeck Funktion Above
                            next = nextpos
                            section = [np.array(crossing_pixel)]
                            section += get_basic_section(next, classified_pixels, back)
                            boundingboxpx = []
                            for i in range(boundingbox[0], boundingbox[0] + boundingbox[2]):
                                for j in range(boundingbox[1], boundingbox[1] + boundingbox[3]):
                                    boundingboxpx.append(np.array([i,j]))
                            if np.any([np.all(section[-1] == bp) for bp in boundingboxpx]) and np.any([np.all(section[0] == bp) for bp in boundingboxpx]):
                                continue

                            return section

    return None


def find_missing(crossing_pixel, sections, classified_pixels):
    radius = 1
    for s in sections:
        s[0][0] = np.array(s[0][0])
    for i in range(-radius, +radius + 1):
        for j in range(-radius, +radius + 1):
            # if i == 0 and j == 0:
            #     continue
            currentpos = (crossing_pixel[0] + i, crossing_pixel[1] + j)
            in_sections = False
            if classified_pixels[currentpos] == 3:
                back = currentpos
                radius2 = 1
                for p in range(-radius2, +radius2 + 1):
                    for q in range(-radius2, +radius2 + 1):
                        if p == 0 and q == 0:
                            continue
                        nextpos = (currentpos[0] + p, currentpos[1] + q)
                        if classified_pixels[nextpos] == 2 and not any(
                                [np.any(np.all(np.array(nextpos) == s[0], axis=1)) for s in sections]):
                            next = nextpos
                            section = [np.array(crossing_pixel)]
                            section += get_basic_section(next, classified_pixels, back)
                            return section
    return None


def get_basic_section(start, classified_pixels, back):
    delta = np.array([0, 0])
    section = []
    section.append(np.array(back))
    section.append(np.array(start))
    x, y = start
    neighbor = None
    neighbor_value = -float('inf')

    for i in range(0, 3):
        for j in range(0, 3):
            if (i != 1 or j != 1) and (x + i - 1 != back[0] or y + j - 1 != back[1]) and (
                    classified_pixels[x + i - 1, y + j - 1] == 2):
                neighbor = np.array([x + i - 1, y + j - 1])
                neighbor_value = classified_pixels[x + i - 1, y + j - 1]

    next = neighbor
    next_value = neighbor_value
    delta = np.subtract(next, start)

    while next_value == 2:  # edge pixel
        section.append(next)
        neighbor = None
        neighbor_value = -float('inf')
        x, y = next
        for i in range(0, 3):
            for j in range(0, 3):
                if (i != 1 or j != 1) and (i != 1 - delta[0] or j != 1 - delta[1]) and (
                        classified_pixels[x + i - 1, y + j - 1] > neighbor_value):
                    neighbor = np.array([x + i - 1, y + j - 1])
                    neighbor_value = classified_pixels[x + i - 1, y + j - 1]
        next_value = neighbor_value
        delta = np.subtract(neighbor, next)
        next = neighbor
    if next_value == 3:
        section.append(next)
    return section


def traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, port_pixels, boundinboxes):
    merged_sections = []

    classified_pixelsf2 = copy.deepcopy(classified_pixels)
    for (crossing_pixel, sections) in crossing_pixels_in_port_sections.items():
        classified_pixelsf1 = copy.deepcopy(classified_pixels)
        found = True
        while found:
            found = find_missing2(crossing_pixel, sections, classified_pixels, boundinboxes[crossing_pixel])
            if not found is None:
                classified_pixelsf = copy.deepcopy(classified_pixels)
                for i in found:
                    classified_pixelsf[i[0], i[1]] = 10
                    classified_pixelsf1[i[0], i[1]] = 10
                    classified_pixelsf2[i[0], i[1]] = 10
                crossing_pixels_in_port_sections[crossing_pixel].append(found)
                found = True
            else:
                found = False
        sectiondirs = []
        for cur_section in sections:
            # if crossing pixel is already visited, then continue
            if len(cur_section) <def_dir_offset:
                if(len(cur_section)<=1):
                    print("Linienabschnitslänge ist zu klein")
                dir_offset = len(cur_section)

            else:
                dir_offset = def_dir_offset  # if crossing not allways at the end set to +/-

            #start_pixel = crossing_pixel
            if np.array_equal(cur_section[0], np.array(crossing_pixel)):
                start_pixel = cur_section[1]
                end_pixel = cur_section[dir_offset]
            else:
                start_pixel = cur_section[-2]
                end_pixel = cur_section[-dir_offset]
            section_dir = end_pixel - start_pixel
            sectiondirs.append([cur_section, section_dir])


        skip_sections = []
        for first_section in sectiondirs:
            angle = []
            if section_in_sections(first_section[0],skip_sections):
                continue
            for other_section in sectiondirs:
                if section_equal(first_section[0], other_section[0]):
                    continue
                angle.append([calcangle(first_section[1], other_section[1]), other_section[0]])
            angle.sort(key=lambda x: x[0], reverse=True)

            # Todo check for Reverse
            skip_sections.append(angle[0][1])
            if np.array_equal(first_section[0][-1], np.array(crossing_pixel)):
                if np.array_equal(angle[0][1][-1], np.array(crossing_pixel)):
                    # ToDo check revese auf fehler
                    first_section[0].reverse()
                    merged_sections.append(angle[0][1] + first_section[0][1:])
                elif np.array_equal(angle[0][1][0], np.array(crossing_pixel)):
                    merged_sections.append(first_section[0][:-1] + angle[0][1])

                else:
                    print("ERORR Traversal subphase 1")
            elif np.array_equal(first_section[0][0], np.array(crossing_pixel)):
                if np.array_equal(angle[0][1][-1], np.array(crossing_pixel)):
                    merged_sections.append(angle[0][1] + first_section[0][1:])
                elif np.array_equal(angle[0][1][0], np.array(crossing_pixel)):
                    first_section[0].reverse()
                    merged_sections.append(first_section[0][:-1] + angle[0][1])
                else:
                    print("ERORR Traversal subphase 2")
            else:
                print("ERORR Traversal subphase 0")
    merged = True
    while merged:
        merged = False
        for s1 in merged_sections:
            # if s1 in skip:
            #     continue
            for s2 in merged_sections:
                # if s2 in skip:
                #     continue
                classified_pixels2 = copy.deepcopy(classified_pixels)
                for i in s2:
                    classified_pixels2[i[0], i[1]] = 10
                classified_pixels1 = copy.deepcopy(classified_pixels)
                for i in s1:
                    classified_pixels1[i[0], i[1]] = 10
                if section_equal(s1, s2):
                    continue
                if not ((s1[0][0], s1[0][1]) in port_pixels and (s1[-1][0], s1[-1][1]) in port_pixels):
                    if not ((s2[0][0], s2[0][1]) in port_pixels and (s2[-1][0], s2[-1][1]) in port_pixels):
                        if any(np.all(s1[-1] == s2, axis=1)) and any(np.all(s1[-2] == s2, axis=1)):
                            w1 = np.where(np.all(s1[-1] == s2, axis=1) == True)[0][0]
                            w2 = np.where(np.all(s1[-2] == s2, axis=1) == True)[0][0]
                            if w1 > w2:
                                s3 = s2[np.where(np.all(s1[-1] == s2, axis=1) == True)[0][0]:]
                                s4 = s1 + s3
                            else:
                                s3 = s2[:np.where(np.all(s1[-1] == s2, axis=1) == True)[0][0]]
                                s3.reverse()
                                s4 = s1 + s3

                            classified_pixels3 = copy.deepcopy(classified_pixels)
                            for i in s3:
                                classified_pixels3[i[0], i[1]] = 10
                            classified_pixels4 = copy.deepcopy(classified_pixels)
                            for i in s4:
                                classified_pixels4[i[0], i[1]] = 10

                            # skip.append(s1)
                            # skip.append(s2)
                            remove_section_from_sections(s1, merged_sections)
                            remove_section_from_sections(s2, merged_sections)
                            # merged_sections.remove(s1)
                            # merged_sections.remove(s2)
                            merged_sections.append(s4)
                            merged = True
                            break
                        elif any(np.all(s1[0] == s2, axis=1)):

                            #skip da s1 und s2 vertauscht sind wird später richtigherum aufgerufen
                            #print("H2")
                            pass
                        else:
                            pass
                            #print("H3")

    return merged_sections


def merge_crossingpixels1(crossing_pixels_in_port_sections, classified_pixels):
    crossing_pixels_in_port_sectionsf = crossing_pixels_in_port_sections.copy()
    for crossing_pixel in crossing_pixels_in_port_sectionsf:
        crossing_pixels_in_range = []
        x, y = crossing_pixel
        for i in range(0, 5):
            for j in range(0, 5):
                if (classified_pixels[x + i - 2, y + j - 2] == 3):
                    crossing_pixels_in_range.append((x + i - 2, y + j - 2))
        if len(crossing_pixels_in_range) > 1:
            cpirnp = np.array(crossing_pixels_in_range)
            minx = int(min(cpirnp[:, 0]))
            maxx = int(max(cpirnp[:, 0]))
            miny = int(min(cpirnp[:, 1]))
            maxy = int(max(cpirnp[:, 1]))
            x1 = int((minx + maxx) / 2)
            y1 = int((miny + maxy) / 2)
            if classified_pixels[x1, y1] == 0:
                classified_pixels[(x1, y1)] = 3
            for cp in crossing_pixels_in_range:
                if not cp in crossing_pixels_in_port_sections:
                    classified_pixels[cp] = 3
                    continue
                if not (x1, y1) in crossing_pixels_in_port_sections:
                    crossing_pixels_in_port_sections[(x1, y1)] = []
                    classified_pixels[(x1, y1)] = 3
                if cp == (x1, y1):
                    continue
                for cpl in crossing_pixels_in_port_sections[cp]:
                    cpl[0].append(np.array([x1, y1]))
                    crossing_pixels_in_port_sections[(x1, y1)].append(cpl)
                crossing_pixels_in_port_sections.pop(cp)

            continue

    return crossing_pixels_in_port_sections


def merge_crossingpixels(crossing_pixels_in_port_sections, classified_pixels):
    all_crossing_pixels = np.stack(np.where(classified_pixels == 3), axis=1)
    for allc in all_crossing_pixels:
        if tuple(allc) not in crossing_pixels_in_port_sections.keys():
            crossing_pixels_in_port_sections[tuple(allc)] = []
    clusters = [[c] for c in crossing_pixels_in_port_sections.items()]
    merge_happened = True
    new_crossing_pixels_in_port_sections = {}
    boundingboxes = {}
    while merge_happened:
        merge_happened = try_to_merge(clusters)
    for c in clusters:
        center = tuple(np.round(np.array([p[0] for p in c]).mean(axis=0)).astype(int))
        boundingrect = cv2.boundingRect(np.asarray([np.asarray(x[0]) for x in c]))
        for x in range(boundingrect[0], boundingrect[0] + boundingrect[2]):
            for y in range(boundingrect[1], boundingrect[1] + boundingrect[3]):
                if classified_pixels[x, y] == 2:
                    classified_pixels[x, y] = 3
        #classified_pixels[center] = 3
        new_sections = []
        for cp in c:
            for cpl in cp[1]:
                if not np.array_equal(cpl[-1], np.array(center)):
                    cpl.append(np.array(center))

            new_sections += cp[1]
        new_crossing_pixels_in_port_sections[center] = new_sections
        boundingboxes[center] = boundingrect
        # a = cv2.circle(a, (center[1],center[0]), 3, 3, -1)
        radius = 3  # smaler cicle = 3
        # for i in range(-radius, +radius + 1):
        #     for j in range(-radius, +radius + 1):
        #         if classified_pixels[center[0]+i, center[1]+j] >0:
        #             classified_pixels[center[0]+i, center[1]+j]=3
        #classified_pixels[center] = 31
    return new_crossing_pixels_in_port_sections, boundingboxes







def euclid_tuple(t1, t2):
    return np.linalg.norm(np.array(tuple(map(lambda i, j: i - j, t1, t2))))




def try_to_merge(clusters):
    thresh = 8.6 # Smaler Circle = 7.1 bigger = 8.5
    for cluster in clusters:
        for crossing_pixel in cluster:
            for other_cluster in clusters:
                if (other_cluster == cluster):
                    continue
                for other_crossing_pixel in other_cluster:
                    if (euclid_tuple(crossing_pixel[0], other_crossing_pixel[0]) < thresh):
                        cluster += other_cluster
                        clusters.remove(other_cluster)
                        return True
    return False


def section_equal(a, b):
    return np.array_equal(np.array(a), np.array(b))


def section_in_sections(section, sections):
    for s in sections:
        if section_equal(s, section):
            return True
    return False

def remove_section_from_sections(section, sections):
    counter = 0
    for s in sections:
        if section_equal(s, section):
            del sections[counter]
        counter+=1
