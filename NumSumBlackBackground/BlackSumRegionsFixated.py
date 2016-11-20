import numpy as np
from keras.models import load_model
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import dilation, disk

model = load_model('model10000.h5')
lines = []
str_elem1 = disk(3)
block_size = [28, 28]

with open("out.txt") as text_file:
    data = text_file.read()
    lines = data.split('\n')

calculated = []
for id, line in enumerate(lines):
    cols = line.split('\t')
    if cols[0] == '':
        continue
    elif id > 1:
        img = imread(cols[0])
        (height, width, c) = img.shape
        img_gray = rgb2gray(img)

        # the numbers are white, the background is black
        img_bin = img_gray > 0.5
        # make sure that every number is classified as a single region
        img_bin_dilation = dilation(img_bin, selem=str_elem1)
        labeled_img = label(img_bin_dilation)
        regions = regionprops(labeled_img)

        # eliminate white noise
        true_regions = []
        for region in regions:
            bbox = region.bbox
            block_height = bbox[2] - bbox[0]
            # during the testing, none of the regions that didn't represent a number were higher than 12
            if block_height > 12:
                true_regions.append(region)

        recognized_numbers = []

        for region in true_regions:
            x = int(round(region.centroid[0]))
            y = int(round(region.centroid[1]))

            # calculate the approximate location of the sliding block by using the centroid field of a region
            # and make sure the block will fit in the image
            block_loc_x = x - block_size[0] / 2
            if block_loc_x > height - block_size[0]:
                block_loc_x = height - block_size[0]
            elif block_loc_x < 0:
                block_loc_x = 0
            block_loc_y = y - block_size[1] / 2
            if block_loc_y > width - block_size[1]:
                block_loc_y = width - block_size[1]
            elif block_loc_y < 0:
                block_loc_y = 0

            block_loc = [block_loc_x, block_loc_y]
            block_img = img_gray[block_loc[0]:block_loc[0] + block_size[0], block_loc[1]:block_loc[1] + block_size[1]]
            block_predict = model.predict(np.array([block_img.flatten()]), verbose=0)

            recognized_numbers.append(block_predict[0].argmax())

        img_info = [cols[0], sum(recognized_numbers)]
        calculated.append(img_info)

with open("out2.txt", 'w') as text_file:
    text_file.write(lines[0] + '\n')
    text_file.write(lines[1] + '\n')
    for id, img_info in enumerate(calculated):
        result = img_info[1]
        text_file.write(img_info[0]+'\t'+"{:.1f}".format(result)+'\n')
