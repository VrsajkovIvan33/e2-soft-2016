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
            x = int(round(region.centroid[0])) - 5
            y = int(round(region.centroid[1])) - 5

            # for every single region find the approximate area where the number is, then slide the block across the
            # area to find the best match from the neural network
            max_value = 0
            max_index = 0
            for i in range(x, x + 10):
                if i + block_size[0] / 2 > height:
                    break
                elif i - block_size[0] / 2 < 0:
                    break
                for j in range(y, y + 10):
                    if j + block_size[1] / 2 > width:
                        break
                    elif j - block_size[1] / 2 < 0:
                        break
                    block_loc = [i - block_size[0] / 2, j - block_size[1] / 2]
                    block_img = img_gray[block_loc[0]:block_loc[0] + block_size[0], block_loc[1]:block_loc[1] + block_size[1]]
                    block_predict = model.predict(np.array([block_img.flatten()]), verbose=0)
                    if np.amax(block_predict[0]) >= max_value:
                        max_value = np.amax(block_predict[0])
                        max_index = block_predict[0].argmax()
            recognized_numbers.append(max_index)

        img_info = [cols[0], sum(recognized_numbers)]
        calculated.append(img_info)

with open("out1.txt", 'w') as text_file:
    text_file.write(lines[0] + '\n')
    text_file.write(lines[1] + '\n')
    for id, img_info in enumerate(calculated):
        result = img_info[1]
        text_file.write(img_info[0]+'\t'+"{:.1f}".format(result)+'\n')
