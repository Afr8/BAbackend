import numpy as np
from circular_list import *
import cv2
import pickle
import math
from matplotlib import pyplot as plt
import os
from collections import Counter
import morphology
import Gates
def start():

    global IMAGE_NAME
    global IMAGE_PATH
    IMAGE_NAME = "98d585a4-e318-4d6d-98c0-8a2226abe9d4"
    IMAGE_PATH = os.path.join("images", IMAGE_NAME + '.jpg')
    detections ,image_np= load(IMAGE_NAME,IMAGE_PATH)
    boxes, imageblack, box_pixel= drawboxes(detections ,image_np)
    pre= preprocessing(imageblack)
    skel= drawedges(pre,boxes)
    classified_pixels, port_pixels= getclassifiedpixels(skel)
    trivial_sections,port_sections, crossing_pixels_in_port_sections, last_gradients1= edge_sections_identify(classified_pixels, port_pixels)
    merged_sections = traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, last_gradients1)
    edge_sections = trivial_sections + merged_sections
    connections = connectNodes(edge_sections, boxes)
    elements = build(boxes,connections, image_np)
    print(str())
    print("Done")

def connectNodes(edge_sections,boxes):
    connections = {}
    counter = 0
    for section in edge_sections:
        if section[0][1] < section[-1][1]:
            connections[counter] = [[section[0], (section[-1][0], section[-1][1])], [0, 0]]
            c = 0
            for box in boxes:

                if box[3] == section[0][1] and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][0] = c
                if box[1] == section[-1][1] + 1 and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][1] = c
                c += 1
        else:
            connections[counter] = [[(section[-1][0], section[-1][1]),section[0]], [0, 0]]
            c = 0
            for box in boxes:

                if box[3] == section[-1][1] and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][0] = c
                if box[1] == section[0][1] + 1 and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][1] = c
                c += 1
        counter += 1

    return connections

def build(boxes, connections, image_np):

    elements = []
    alt = 0
    for box in boxes:

        if box[4] == 0:
            #print("switch_True")
            elements.append(Gates.Input(True))
        elif box[4] == 1:
            #print("switch_False")
            elements.append(Gates.Input())
        elif box[4] == 2:
            #print("and")
            elements.append(Gates.And())
        elif box[4] == 3:
            #print("or")
            elements.append(Gates.Or())
        elif box[4] == 4:
            #print("nand")
            elements.append(Gates.Nand())
            #elements[-1].addInput(alt)
        elif box[4] == 5:
            #print("nor")
            elements.append(Gates.Nor())
        elif box[4] == 6:
            #print("not")
            elements.append(Gates.Not())
        elif box[4] == 7:
            #print("bulb")
            elements.append(Gates.Output())
        alt +=1
    #print("con")

    for connection in connections.values():
        print("Connect: " + str(connection[1][0]) + " mit " + str(connection[1][1]))
        if not type(elements[connection[1][1]]) == Gates.Input:
            elements[connection[1][1]].addInput(elements[connection[1][0]])



    for element in elements:
        if type(element) == Gates.Input:
            element.update()
    print("Lösung")
    count =0
    for element in elements:
        if type(element) == Gates.Output:
            print(element.getValue())
            cv2.putText(
                image_np,  # numpy array on which text is written
                str(element.getValue()),  # text
                (int((boxes[count][1] + boxes[count][3]) / 2 -20), int((boxes[count][0] + boxes[count][2]) / 2 - 40)),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,  # font family
                1,  # font size
                (0, 0, 0),  # font color
                3)  # font stroke
        count += 1
    cv2.imwrite(os.path.join('images', IMAGE_NAME + '_Solution' + '.jpg'), image_np)
    return elements







def load(IMAGE_NAME, IMAGE_PATH):
    with open("images/"+ IMAGE_NAME + "_detections.pickle", "rb") as file:
        detections = pickle.load(file)
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    return detections , image_np

def drawboxes(detections,image_np):
    count = 0
    for i in detections['detection_scores']:
        if i >= 0.75:
            count += 1
    boxes = np.zeros((count,5))
    box_pixel = np.zeros((image_np.shape[0], image_np.shape[1], image_np.shape[2]))
    imageblack = image_np.copy()
    image_np2 = image_np.copy()
    for q in range(0, count):
        xmin = detections['detection_boxes'][q][0] * image_np.shape[0] - 2
        ymin = detections['detection_boxes'][q][1] * image_np.shape[1] - 2
        xmax = detections['detection_boxes'][q][2] * image_np.shape[0] + 2
        ymax = detections['detection_boxes'][q][3] * image_np.shape[1] + 2
        print(xmin)

        cv2.putText(
            image_np2, # numpy array on which text is written
            str(q),  # text
            (int((ymin +ymax )/2),int((xmin +xmax )/2-40)),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            1,  # font size
            (0, 0, 0),  # font color
            3)  # font stroke
        boxes[q] = [int(xmin), int(ymin), int(xmax), int(ymax), detections['detection_classes'][q]]
        imageblack[int(xmin):int(xmax), int(ymin):int(ymax)] = (0, 0, 0)
        box_pixel[int(xmin):int(xmax), int(ymin):int(ymax)] = (255, 255, 255)
    cv2.imwrite(os.path.join('images', IMAGE_NAME + '_imageblack' +'.jpg'), imageblack)
    cv2.imwrite(os.path.join('images', IMAGE_NAME + '_box_pixel' + '.jpg'),box_pixel)
    cv2.imwrite(os.path.join('images', IMAGE_NAME + '_numbers' + '.jpg'), image_np2)
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
    cv2.imwrite(os.path.join('images', IMAGE_NAME +  '_preprocess' + '.jpg'), img2)
    return img2

def drawedges(img2,boxes):
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
                            A[x + 2, y, 0] == 0 or A[x + 2, y - 1, 0] == 1 or A[x + 1, y - 1, 0] == 1)
                d_4 = (y < Kanten.shape[1] - 2) and (
                            A[x - 1, y + 2, 0] == 0 or A[x, y + 2, 0] == 1 or A[x - 1, y + 1, 0] == 1)

                if A[x - 1, y + 1, 0] == 1 and (A[x - 1, y, 0] == 1 and d_1):
                    A[x - 1, y, 0] = 0
                    A[x - 1, y, 1] = 0
                    A[x - 1, y, 2] = 0
                if A[x - 1, y - 1, 0] == 1 and (A[x, y - 1, 0] == 1 and d_2):
                    A[x, y - 1, 0] = 0
                    A[x, y - 1, 1] = 0
                    A[x, y - 1, 2] = 0
                if A[x + 1, y - 1, 0] == 1 and (A[x + 1, y, 0] == 1 and d_3):
                    A[x + 1, y, 0] = 0
                    A[x + 1, y, 1] = 0
                    A[x + 1, y, 2] = 0
                if A[x + 1, y + 1, 0] == 1 and (A[x, y + 1, 0] == 1 and d_4):
                    A[x, y + 1, 0] = 0
                    A[x, y + 1, 1] = 0
                    A[x, y + 1, 2] = 0

    skel = A
    skel = skel * 255
    cv2.imwrite(os.path.join('images', IMAGE_NAME + '_skel' + '.jpg'), skel)
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






def edge_sections_identify(classified_pixels,port_pixels):
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
                        if (i != 1 or j != 1) and (i != 1-delta[0] or j!= 1-delta[1]) and (classified_pixels[x + i - 1, y + j - 1] > neighbor_value):
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


    bild = np.zeros((classified_pixels.shape[0], classified_pixels.shape[1], 3))
    bild2= np.zeros((classified_pixels.shape[0],classified_pixels.shape[1],3))
    for x in range(0,len(trivial_sections)):
        for z in trivial_sections[x]:
            bild[z[0],z[1],0] = 255
            bild[z[0], z[1], 1] = 255
            bild[z[0], z[1], 2] = 255
    for x in range(0,len(port_sections)):
        for z in port_sections[x]:
            bild2[z[0],z[1],0] = 255
            bild2[z[0], z[1], 1] = 255
            bild2[z[0], z[1], 2] = 255
    return trivial_sections,port_sections, crossing_pixels_in_port_sections, last_gradients1









def traversal_subphase(classified_pixels, crossing_pixels_in_port_sections, last_gradients):
    merged_sections = []

    for crossing_pixel in crossing_pixels_in_port_sections:
        for info_section in crossing_pixels_in_port_sections[crossing_pixel]:

            # if crossing pixel is already visited, then continue
            if info_section[1] == 1:
                continue

            section = info_section[0]
            start_pixel = crossing_pixel

            flag_found_section = False
            iteration = 0

            while not flag_found_section:

                crossing_section_direction = get_crossing_section_direction(classified_pixels, start_pixel,
                                                                            last_gradients[section[0]], section)

                flag_found_section = merge_sections(crossing_pixels_in_port_sections, section,
                                                    crossing_section_direction, merged_sections)

                if not flag_found_section:

                    if len(crossing_section_direction) > 1:
                        # start_back is a crossing pixel
                        start_back = crossing_section_direction[-2]
                    else:
                        start_back = start_pixel

                    # next is an edge pixel
                    next = crossing_section_direction[-1]

                    _section, _last_gradients, next = get_basic_section(next, classified_pixels, start_back)
                    crossing_section_direction.extend(_section[1:])

                    _crossing_section_direction = get_crossing_section_direction(classified_pixels, _section[-1],
                                                                                 last_gradients[section[0]], _section)
                    crossing_section = crossing_section_direction + _crossing_section_direction

                    flag_found_section = merge_sections(crossing_pixels_in_port_sections, section, crossing_section,
                                                        merged_sections)

                    if not flag_found_section:
                        # start pixel is a crossing pixel
                        start_pixel = crossing_section[-2]
                        section = section + crossing_section

                    # if iteration == 1:
                    #	_last_gradients.clear()
                    #	del _last_gradients
                    # else:
                    last_gradients[section[0]].extend(_last_gradients)

                iteration += 1

    return merged_sections


def merge_sections(crossing_pixels_in_port_sections, section, crossing_section, merged_sections):
    if len(crossing_section) > 1:
        # back is a crossing pixel
        back = crossing_section[-2]
    else:
        back = section[-1]

    key_back = (back[0], back[1])

    # next is an edge pixel
    next = crossing_section[-1]

    if not key_back in crossing_pixels_in_port_sections.keys():
        return False

    for info_section in crossing_pixels_in_port_sections[key_back]:
        _section = info_section[0]

        if next[0] == _section[-2][0] and next[1] == _section[-2][1]:
            merged_sections.append(section + crossing_section + _section[::-1][1:])

            # mark back (crossing pixel) as already visited
            info_section[1] = 1

            # print ((section[0][0], section[0][1]), (crossing_section[0][0], crossing_section[0][1]), (crossing_section[-1][0], crossing_section[-1][1]), (_section[::-1][-1][0], _section[::-1][-1][1])), next

            return True

    return False

def get_basic_section(start, classified_pixels, start_back=None):
    # 'gradient' vector
    delta = np.array([0, 0])

    last_gradients = Circular_list()

    section = []
    section.append(start)

    x, y = start
    next, next_value = get_max_neighbor(classified_pixels, x, y, start_back)
    delta = np.subtract(next, start)

    while next_value == 2:  # edge pixel

        last_gradients.insert((delta[0], delta[1]))
        section.append(next)

        next = np.add(next, delta)
        next_value = classified_pixels[next[0], next[1]]

        if next_value < 2:  # blank pixel or miscellaneous pixel
            last = section[-1]  # get last element added in section
            x, y = last
            back = np.subtract(last, delta)

            # get max value in the neighborhood, unless the 'back'
            next, next_value = get_max_neighbor(classified_pixels, x, y, back)

            delta = np.subtract(next, last)

    last_gradients.insert((delta[0], delta[1]))
    section.append(next)
    last_element = next

    return section, last_gradients, last_element


def get_max_neighbor(classified_pixels, x, y, back=None):
    neighbor = None
    neighbor_value = -float('inf')

    for i in range(0, 3):
        for j in range(0, 3):

            if (back is None or (x + i - 1 != back[0] or y + j - 1 != back[1])) and (i != 1 or j != 1) and (
                    classified_pixels[x + i - 1, y + j - 1] > neighbor_value):
                neighbor = np.array([x + i - 1, y + j - 1])
                neighbor_value = classified_pixels[x + i - 1, y + j - 1]

    return neighbor, neighbor_value


def get_crossing_section_direction(classified_pixels, crossing_pixel, last_gradients, section):
    # counter gradients frequency
    cnt_gradient = Counter(last_gradients.get_list())
    # count in list
    grads = cnt_gradient.most_common()

    crossing_section_direction = []

    next = crossing_pixel
    next_value = classified_pixels[next[0], next[1]]

    # back is a edge pixel
    back = section[-2][0], section[-2][1]

    # avoid local minima
    iterations = 0
    loop_grads = Circular_list(3)
    excluded_grad = None

    while next_value != 2:  # edge pixel

        aux_value = 0
        i = 0

        if iterations == 3:
            list_loop_grads = loop_grads.get_list()
            excluded_grad = list_loop_grads[1]
            crossing_section_direction[:] = []
            iterations = 0


        while aux_value < 2 and i < len(grads):  # blank pixel or miscellaneous and i < len

            if grads[i][0] == excluded_grad:
                continue

            delta = grads[i][0]
            aux = np.add(next, delta)

            if aux[0] == back[0] and aux[1] == back[1]:  # back[0] >= 0 and back[1] >= 0 and
                aux_value = 0
            else:
                aux_value = classified_pixels[aux[0], aux[1]]

            i += 1

        if aux_value < 2 and i == len(grads):
            delta = get_gradient(classified_pixels, back, next, grads, excluded_grad)
            loop_grads.insert(delta)
            back = next[0], next[1]
            next = np.add(next, delta)
            next_value = classified_pixels[next[0], next[1]]

        else:
            loop_grads.insert(delta)
            back = next[0], next[1]
            next = aux
            next_value = aux_value

        crossing_section_direction.append(next)
        iterations += 1

    return crossing_section_direction


def get_gradient(classified_pixels, back, current, common_grads, excluded_grad=None):
    possible_grads = {(0, 1), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1), (0, -1), (-1, 0)}
    s_grads = [x[0] for x in common_grads]
    possible_grads = possible_grads - set(s_grads)

    if not excluded_grad is None:
        possible_grads = possible_grads - {excluded_grad}

    min_d = float('inf')
    min_grad = None

    for grad in possible_grads:
        aux = np.add(current, grad)

        sat_condition = (aux[0] != back[0] or aux[1] != back[1]) and (classified_pixels[aux[0], aux[1]] > 1)

        d = distance_heuristic_grads(common_grads, possible_grads, grad)

        if sat_condition and (d < min_d):
            min_d = d
            min_grad = grad

    return min_grad

def distance_heuristic_grads(common_grads, possible_grads, grad):
    n = 0.0
    average_common_grad = [0.0, 0.0]
    # most_common_grad = grads[0][0]

    for _grad in common_grads:
        aux = [_grad[1] * z for z in _grad[0]]
        average_common_grad = map(sum, zip(average_common_grad, aux))
        n += _grad[1]

    average_common_grad = [z / n for z in average_common_grad]

    # determine weights to calculate distances
    amount_non_zero_x = 0
    amount_non_zero_y = 0

    for _grad in common_grads:
        if _grad[0][0] != 0:
            amount_non_zero_x += _grad[1]
        if _grad[0][1] != 0:
            amount_non_zero_y += _grad[1]

    total_non_zeros = amount_non_zero_x + amount_non_zero_y

    alpha = (total_non_zeros - amount_non_zero_y)
    betha = (total_non_zeros - amount_non_zero_x)

    # print alpha, betha

    d = weighted_euclidean_distance(average_common_grad, grad, alpha, betha)
    # d = weighted_euclidean_distance(most_common_grad, grad, alpha, betha)
    # print amount_non_zero_x, amount_non_zero_y, "alpha: ", alpha, "betha: ", betha, "distance: ", d

    return d

def weighted_euclidean_distance(grad1, grad2, alpha=1, betha=1):
	[x, y] = [grad1[0] - grad2[0], grad1[1] - grad2[1]]
	d = np.sqrt((alpha* (x**2)) + (betha * (y**2)))
	return d