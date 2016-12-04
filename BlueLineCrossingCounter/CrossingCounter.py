import numpy as np
import cv2

with open("out.txt") as file:
    data = file.read()
    lines = data.split('\n')

calculated = []
for id, line in enumerate(lines):
    cols = line.split('\t')
    if (cols[0] == ''):
        continue
    elif (id > 1):
        cap = cv2.VideoCapture(cols[0])

        width_lower = []
        width_upper = []
        length_lower = []
        length_upper = []
        target_contour = None
        found_contours = []
        number_of_crossings = 0

        # int(cap.get(7)) is the number of frames in a video
        for j in range(0, int(cap.get(7))):
            ret, frame = cap.read()
            # use only every 60th frame, in order to boost the performance
            if ret is True and j % 60 == 0:

                # for the first frame, find the blue line
                if j == 0:
                    img = frame
                    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 10, 255, 0)
                    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # find the blue line
                    max_area = 0.0
                    target_contour = None
                    for contour in contours:
                        # find the biggest contour, but don't count the gray counter in the upper left corner
                        if cv2.contourArea(contour) > max_area and cv2.contourArea(contour) < 7500.0:
                            max_area = cv2.contourArea(contour)
                            target_contour = contour

                    # get the four points of the minimal bounding rectangle
                    rect = cv2.minAreaRect(target_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # pair the points
                    width1 = []
                    width2 = []
                    for index, point in enumerate(box):
                        if index == 0:
                            width1.append(point)
                        elif len(width1) == 2:
                            width2.append(point)
                        elif len(width2) == 2:
                            width1.append(point)
                        elif width1[0][0] - point[0] < 50:
                            width1.append(point)
                        else:
                            width2.append(point)

                    # calculate the line functions that make the widths
                    line1 = []
                    line1.append(float(width1[1][1] - width1[0][1]) / float(width1[1][0] - width1[0][0]))
                    line1.append(width1[0][1] - line1[0] * width1[0][0])

                    line2 = []
                    line2.append(float(width2[1][1] - width2[0][1]) / float(width2[1][0] - width2[0][0]))
                    line2.append(width2[0][1] - line2[0] * width2[0][0])

                    # create a larger area surrounding the blue line
                    widening = 12

                    new_width1 = []
                    if width1[0][0] > width1[1][0]:
                        new_width1.append([width1[0][0] + widening, line1[0] * (width1[0][0] + widening) + line1[1]])
                        new_width1.append([width1[1][0] - widening, line1[0] * (width1[1][0] - widening) + line1[1]])
                    else:
                        new_width1.append([width1[0][0] - widening, line1[0] * (width1[0][0] - widening) + line1[1]])
                        new_width1.append([width1[1][0] + widening, line1[0] * (width1[1][0] + widening) + line1[1]])

                    new_width2 = []
                    if width2[0][0] > width2[1][0]:
                        new_width2.append([width2[0][0] + widening, line2[0] * (width2[0][0] + widening) + line2[1]])
                        new_width2.append([width2[1][0] - widening, line2[0] * (width2[1][0] - widening) + line2[1]])
                    else:
                        new_width2.append([width2[0][0] - widening, line2[0] * (width2[0][0] - widening) + line2[1]])
                        new_width2.append([width2[1][0] + widening, line2[0] * (width2[1][0] + widening) + line2[1]])

                    new_box = []
                    new_box.append(new_width1[0])
                    new_box.append(new_width1[1])
                    new_box.append(new_width2[0])
                    new_box.append(new_width2[1])
                    new_box = np.int0(new_box)
                    cv2.drawContours(img, [new_box], 0, (0, 255, 0), 2)

                    # calculate the line functions that surround the area
                    min_width1 = None
                    max_width1 = None
                    if new_width1[0][0] < new_width1[1][0]:
                        min_width1 = new_width1[0]
                        max_width1 = new_width1[1]
                    else:
                        min_width1 = new_width1[1]
                        max_width1 = new_width1[0]

                    min_width2 = None
                    max_width2 = None
                    if new_width2[0][0] < new_width2[1][0]:
                        min_width2 = new_width2[0]
                        max_width2 = new_width2[1]
                    else:
                        min_width2 = new_width2[1]
                        max_width2 = new_width2[0]

                    if min_width1[0] < min_width2[0]:
                        width_lower = line1
                        width_upper = line2
                    else:
                        width_lower = line2
                        width_upper = line1

                    line3 = []
                    line3.append(float(max_width1[1] - max_width2[1]) / float(max_width1[0] - max_width2[0]))
                    line3.append(max_width1[1] - line3[0] * max_width1[0])

                    line4 = []
                    line4.append(float(min_width1[1] - min_width2[1]) / float(min_width1[0] - min_width2[0]))
                    line4.append(min_width1[1] - line4[0] * min_width1[0])

                    length_lower = line3
                    length_upper = line4

                # now that we have the area in which to look for the characters, find and count them in every frame
                img_test = frame
                imgray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
                ret_test, thresh_test = cv2.threshold(imgray_test, 127, 255, 0)
                _, contours_test, hierarchy_test = cv2.findContours(thresh_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour_test in contours_test:
                    # disregard the noise
                    if cv2.contourArea(contour_test) < 20:
                        continue

                    x, y, w, h = cv2.boundingRect(contour_test)
                    rectangle_points = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]

                    # if any of the points of the bounding rectangle is inside the area, the contour will cross the line
                    for point in rectangle_points:
                        if point[1] >= (width_upper[0] * point[0] + width_upper[1]) and point[1] >= (
                                        length_upper[0] * point[0] + length_upper[1]) and point[1] <= (
                                        width_lower[0] * point[0] + width_lower[1]) and point[1] <= (
                                        length_lower[0] * point[0] + length_lower[1]):
                            # first check if the contour has already been in the area in one of the earlier frames
                            already_crossed = False
                            for found_one in found_contours:
                                if cv2.contourArea(found_one) == cv2.contourArea(contour_test):
                                    already_crossed = True
                                    break
                            if already_crossed is False:
                                number_of_crossings += 1
                                found_contours.append(contour_test)
                            break

        cap.release()
        print number_of_crossings
        vid_info = [cols[0], number_of_crossings]
        calculated.append(vid_info)

with open("out1.txt", 'w') as text_file:
    text_file.write(lines[0] + '\n')
    text_file.write(lines[1] + '\n')
    for id, vid_info in enumerate(calculated):
        text_file.write(vid_info[0] + '\t' + str(vid_info[1]) + '\n')
