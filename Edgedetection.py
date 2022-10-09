import copy
import math
import os
import pickle

import cv2
import numpy as np

import Gates
import morphology
from circular_list import *


def start():
    global IMAGE_NAME
    global IMAGE_PATH
    IMAGE_NAME = "0e49e5c8-2c3e-4c0e-90fc-da37e47711d4"
    print(IMAGE_NAME)
    IMAGE_PATH = os.path.join("images", IMAGE_NAME)
    detections, image_np = load(IMAGE_NAME, IMAGE_PATH)
    boxes, imageblack, box_pixel = drawboxes(detections, image_np)
    pre = preprocessing(imageblack)
    skel = drawedges(pre, boxes)
    classified_pixels, port_pixels = getclassifiedpixels(skel)
    trivial_sections, port_sections, crossing_pixels_in_port_sections, last_gradients1 = edge_sections_identify(
        classified_pixels, port_pixels)
    crossing_pixels_in_port_sections = merge_crossingpixels1(crossing_pixels_in_port_sections, classified_pixels)
    merged_sections = traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, port_pixels)
    edge_sections = trivial_sections + merged_sections
    classified_pixelsa = copy.deepcopy(classified_pixels)
    for a in edge_sections:
        for b in a:
            classified_pixelsa[b[0], b[1]] = 10
        classified_pixelsa = copy.deepcopy(classified_pixels)
    connections = connectNodes(edge_sections, boxes)
    elements = build(boxes, connections)
    solve(elements, boxes, image_np)
    Tabelle(elements)
    return elements


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
            switched = inputelements[scounter][0].switch()
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


def connectNodes(edge_sections, boxes):
    connections = {}
    counter = 0
    for section in edge_sections:
        if section[0][1] < section[-1][1]:
            connections[counter] = [[section[0], (section[-1][0], section[-1][1])], [-1, -1]]
            c = 0
            for box in boxes:

                if (box[3] == section[0][1] or box[3] + 1 == section[0][1]) and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][0] = c
                if box[1] == section[-1][1] + 1 and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][1] = c
                c += 1
        else:
            connections[counter] = [[(section[-1][0], section[-1][1]), section[0]], [-1, -1]]
            c = 0
            for box in boxes:

                if (box[3] == section[-1][1] or box[3] + 1 == section[-1][1]) and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][0] = c
                if box[1] == section[0][1] + 1 and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][1] = c
                c += 1
        counter += 1
    return connections


def build(boxes, connections):
    elements = []
    alt = 0
    for box in boxes:

        if box[4] == 0:
            # print("switch_True")
            elements.append([Gates.Input(True), alt])
        elif box[4] == 1:
            # print("switch_False")
            elements.append([Gates.Input(), alt])
        elif box[4] == 2:
            # print("and")
            elements.append([Gates.And(), alt])
        elif box[4] == 3:
            # print("or")
            elements.append([Gates.Or(), alt])
        elif box[4] == 4:
            # print("nand")
            elements.append([Gates.Nand(), alt])
            # elements[-1].addInput(alt)
        elif box[4] == 5:
            # print("nor")
            elements.append([Gates.Nor(), alt])
        elif box[4] == 6:
            # print("not")
            elements.append([Gates.Not(), alt])
        elif box[4] == 7:
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
            cv2.putText(
                image_np,  # numpy array on which text is written
                str(element[0].getValue()),  # text
                (int((boxes[count][1] + boxes[count][3]) / 2 - 20), int((boxes[count][0] + boxes[count][2]) / 2 - 40)),
                # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,  # font family
                1,  # font size
                (0, 0, 0),  # font color
                3)  # font stroke
        count += 1
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_Solution' + '.jpg'), image_np)


def load(IMAGE_NAME, IMAGE_PATH):
    with open("images/" + IMAGE_NAME + "/" + IMAGE_NAME + "_detections.pickle", "rb") as file:
        detections = pickle.load(file)
    img = cv2.imread(IMAGE_PATH + "/" + IMAGE_NAME + ".jpg")
    image_np = np.array(img)

    return detections, image_np


def drawboxes(detections, image_np):
    count = 0
    for i in detections['detection_scores']:
        if i >= 0.75:
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
            image_np2,  # numpy array on which text is written
            str(q),  # text
            (int((ymin + ymax) / 2), int((xmin + xmax) / 2 - 40)),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            1,  # font size
            (0, 0, 0),  # font color
            3)  # font stroke
        boxes[q] = [int(xmin), int(ymin), int(xmax), int(ymax), detections['detection_classes'][q]]
        imageblack[int(xmin):int(xmax), int(ymin):int(ymax)] = (0, 0, 0)
        box_pixel[int(xmin):int(xmax), int(ymin):int(ymax)] = (255, 255, 255)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_imageblack' + '.jpg'), imageblack)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_box_pixel' + '.jpg'), box_pixel)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_numbers' + '.jpg'), image_np2)
    return boxes, imageblack, box_pixel


def preprocessing(imageblack):
    imgray = cv2.cvtColor(imageblack, cv2.COLOR_BGR2GRAY)
    otsu_threshold, image_result = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img2 = np.zeros((imageblack.shape[0], imageblack.shape[1], imageblack.shape[2]))
    img2[:, :, 0] = image_result
    img2[:, :, 1] = image_result
    img2[:, :, 2] = image_result
    img2 = 255 - img2
    img2 = img2.astype(np.uint64)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_preprocess' + '.jpg'), img2)
    return img2


def drawedges(img2, boxes):
    Kanten = np.copy(img2)

    for q in boxes:
        Kanten[int(q[0]):int(q[2]), int(q[1]):int(q[3])] = (0, 0, 0)
    A = np.copy(Kanten) / 255
    A = morphology.zhang_and_suen_binary_thinning(A)

    for x in range(1, Kanten.shape[0] - 1):
        for y in range(1, Kanten.shape[1] - 1):
            if A[x, y, 0] == 1:

                d_1 = (x > 2) and (A[x - 2, y - 1, 0] == 0 or A[x - 2, y, 0] == 1 or A[x - 1, y - 1, 0] == 1)
                d_2 = (y > 2) and (A[x + 1, y - 2, 0] == 0 or A[x, y - 2, 0] == 1 or A[x + 1, y - 1, 0] == 1)
                d_3 = (y < Kanten.shape[1] - 2) and (
                        A[x - 1, y + 2, 0] == 0 or A[x, y + 2, 0] == 1 or A[x - 1, y + 1, 0] == 1)
                d_4 = (y < Kanten.shape[1] - 2) and (
                        A[x + 1, y + 2, 0] == 0 or A[x, y + 2, 0] == 1 or A[x + 1, y + 1, 0] == 1)

                if A[x - 1, y + 1, 0] == 1 and (A[x - 1, y, 0] == 1 and d_1):
                    A[x - 1, y, 0] = 0
                    A[x - 1, y, 1] = 0
                    A[x - 1, y, 2] = 0
                if A[x - 1, y - 1, 0] == 1 and (A[x, y - 1, 0] == 1 and d_2):
                    A[x, y - 1, 0] = 0
                    A[x, y - 1, 1] = 0
                    A[x, y - 1, 2] = 0
                if A[x + 1, y + 1, 0] == 1 and (A[x, y + 1, 0] == 1 and d_3):
                    A[x + 1, y, 0] = A[x, y + 1, 0] = 0
                    A[x + 1, y, 1] = A[x, y + 1, 1] = 0
                    A[x + 1, y, 2] = A[x, y + 1, 2] = 0
                if A[x - 1, y + 1, 0] == 1 and (A[x, y + 1, 0] == 1 and d_4):
                    A[x, y + 1, 0] = 0
                    A[x, y + 1, 1] = 0
                    A[x, y + 1, 2] = 0

    skel = A

    skel = skel * 255
    # skel[294,973]= [255.0,255.0,255.0]

    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_skel' + '.jpg'), skel)
    return skel


def getclassifiedpixels(skel):
    skel = skel / 255
    height = skel.shape[0]
    width = skel.shape[1]

    classified_pixels = np.zeros((height, width))
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
    port_sections = []
    crossing_sections = []
    start_pixels = {}
    start_pixels = dict.fromkeys(port_pixels, 0)
    last_gradients1 = {}
    crossing_pixels_in_port_sections = {}

    for start in start_pixels:
        # if port pixel is already visited, then continue
        if start_pixels[start] == 1:
            continue
        else:

            start_pixels[start] = 1
            delta = np.array([0, 0])
            last_gradients = Circular_list()
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

                last_gradients.insert((delta[0], delta[1]))

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

            last_gradients.insert((delta[0], delta[1]))

            section.append(next)
            last_element = next
            last_gradients1[start] = last_gradients
            next = last_element
        next_value = classified_pixels[next[0], next[1]]
        if next_value == 4:  # port pixel

            # marks the next pixel as already visited
            start_pixels[(next[0], next[1])] = 1
            trivial_sections.append(section)

        elif next_value == 3:  # crossing pixel
            port_sections.append(section)
            pos = (next[0], next[1])
            if not pos in crossing_pixels_in_port_sections:
                crossing_pixels_in_port_sections[pos] = []
            crossing_pixels_in_port_sections[pos].append([section, 0])
    start_pixels.clear()

    return trivial_sections, port_sections, crossing_pixels_in_port_sections, last_gradients1


def calcangle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)


def find_missing2(crossing_pixel, sections, classified_pixels):
    radius = 1
    for s in sections:
        s[0][0] = np.array(s[0][0])
    # todo radius kann außerhalb des bildes enden
    #while found


    radiust = 3
    for i in range(-radiust, +radiust + 1):
        for j in range(-radiust, +radiust + 1):
            if ( radiust > i and i > -radiust) and (radiust > j and j > -radiust):
                continue
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


def find_missing(crossing_pixel, sections, classified_pixels):
    radius = 1
    for s in sections:
        s[0][0] = np.array(s[0][0])
    # todo radius kann außerhalb des bildes enden
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


# todo replace old get_basic_section
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


def traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, port_pixels):
    merged_sections = []
    dir_offset = 4
    classified_pixelsf2 = copy.deepcopy(classified_pixels)
    anter = 0
    for (crossing_pixel, sections) in crossing_pixels_in_port_sections.items():
        classified_pixelsf1 = copy.deepcopy(classified_pixels)
        anter += 1
        found = True
        while found:
            found = find_missing2(crossing_pixel, sections, classified_pixels)
            if not found is None:
                classified_pixelsf = copy.deepcopy(classified_pixels)
                for i in found:
                    classified_pixelsf[i[0], i[1]] = 10
                    classified_pixelsf1[i[0], i[1]] = 10
                    classified_pixelsf2[i[0], i[1]] = 10
                crossing_pixels_in_port_sections[crossing_pixel].append([found, 0])
                found = True
            else:
                found = False
            # todo check found part of allready merged_sections
            #   replace found by merged
            #   zu crossing_pixels_in_port_sections hinzufügen
            #   beachte nach weiter mergen nicht neu zu mergedsection hinzufügen sondern bereits gemeregdte section anpassen
        sectiondirs = []
        for cur_section in sections:
            # if crossing pixel is already visited, then continue

            dir_offset = -4  # if crossing not allways at the end set to +/-

            cur_section = cur_section[0]
            start_pixel = crossing_pixel
            if np.array_equal(cur_section[0], np.array(crossing_pixel)):
                end_pixel = cur_section[-dir_offset]
            else:
                end_pixel = cur_section[dir_offset]
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
                    # todo check for right
                elif np.array_equal(angle[0][1][0], np.array(crossing_pixel)):
                    merged_sections.append(first_section[0][:-1] + angle[0][1])
                    # todo check for right
                else:
                    print("ERORR Traversal subphase 1")
            elif np.array_equal(first_section[0][0], np.array(crossing_pixel)):
                if np.array_equal(angle[0][1][-1], np.array(crossing_pixel)):
                    merged_sections.append(first_section[0][:-1] + angle[0][1])
                    # todo check for right
                elif np.array_equal(angle[0][1][0], np.array(crossing_pixel)):
                    # ToDo check revese auf fehler
                    first_section[0].reverse()
                    merged_sections.append(first_section[0][:-1] + angle[0][1])
                    # todo check for right
                else:
                    print("ERORR Traversal subphase 2")
            else:
                print("ERORR Traversal subphase 0")
    # ToDo While Loop
    for _ in range(0,10):
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
                        if any(np.all(s1[-1] == s2, axis=1)):
                            w1 = np.where(np.all(s1[-1] == s2, axis=1) == True)[0][0]
                            w2 = np.where(np.all(s1[-5] == s2, axis=1) == True)[0][0]
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
                            break
                        elif any(np.all(s1[0] == s2, axis=1)):

                            #skip da s1 und s2 vertauscht sind wird später richtigherum aufgerufen
                            #print("H2")
                            pass
                        else:
                            pass
                            #print("H3")

    return merged_sections


def merge_crossingpixels(crossing_pixels_in_port_sections, classified_pixels):
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
                    # Todo if cp immer am ende von CPL ist alles gut sonst ändern
                    cpl[0].append(np.array([x1, y1]))
                    crossing_pixels_in_port_sections[(x1, y1)].append(cpl)
                crossing_pixels_in_port_sections.pop(cp)

            continue

    return crossing_pixels_in_port_sections


def merge_crossingpixels1(crossing_pixels_in_port_sections, classified_pixels):
    all_crossing_pixels = np.stack(np.where(classified_pixels == 3), axis=1)
    a = copy.deepcopy(classified_pixels)

    for allc in all_crossing_pixels:
        if tuple(allc) not in crossing_pixels_in_port_sections.keys():
            crossing_pixels_in_port_sections[tuple(allc)] = []
    clusters = [[c] for c in crossing_pixels_in_port_sections.items()]
    merge_happened = True
    new_crossing_pixels_in_port_sections = {}
    while merge_happened:
        merge_happened = try_to_merge(clusters)
    for c in clusters:
        center = tuple(np.round(np.array([p[0] for p in c]).mean(axis=0)).astype(int))
        classified_pixels[center] = 3
        new_sections = []
        for cp in c:
            for cpl in cp[1]:
                if not np.array_equal(cpl[0][-1], np.array(center)):
                    cpl[0].append(np.array(center))

            new_sections += cp[1]
        new_crossing_pixels_in_port_sections[center] = new_sections
        #a = cv2.circle(a, (center[1],center[0]), 3, 3, -1)
        radius = 3
        for i in range(-radius, +radius + 1):
            for j in range(-radius, +radius + 1):
                if classified_pixels[center[0]+i, center[1]+j] >0:
                    classified_pixels[center[0]+i, center[1]+j]=3

    return new_crossing_pixels_in_port_sections

def euclid_tuple(t1, t2):
    return np.linalg.norm(np.array(tuple(map(lambda i, j: i - j, t1, t2))))


def try_to_merge(clusters):
    thresh = 7.1
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
