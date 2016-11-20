import numpy as np
from keras.models import load_model
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.measure import regionprops

model = load_model('model10000.h5')
lines = []
block_size = (28, 28)
block_center = (14, 14)

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

        # create a heatmap for every digit
        heatmaps = []
        for i in range(0, 10):
            heatmaps.append(np.zeros((height-28, width-28)))

        # move the sliding block across the image and calculate the heatmap for every digit
        for i in range(14, height-14):
            for j in range(14, width-14):
                block_loc = (i - block_size[0] / 2, j - block_size[1] / 2)
                block_img = img_gray[block_loc[0]:block_loc[0] + block_size[0], block_loc[1]:block_loc[1] + block_size[1]]
                block_predict = model.predict(np.array([block_img.flatten()]), verbose=0)

                for index, arousal in enumerate(block_predict[0]):
                    heatmaps[index][i - 14, j - 14] = arousal

        # calculate the number of appearances for every digit in the image
        recognized_numbers = []
        for number, heatmap in enumerate(heatmaps):
            bin_heatmap = heatmap > 0.95
            lab_heatmap = label(bin_heatmap)
            regions = regionprops(lab_heatmap)
            for k in range(0, len(regions)):
                recognized_numbers.append(number)

        img_info = [cols[0], sum(recognized_numbers)]
        calculated.append(img_info)

with open("out3.txt", 'w') as text_file:
    text_file.write(lines[0] + '\n')
    text_file.write(lines[1] + '\n')
    for id, img_info in enumerate(calculated):
        result = img_info[1]
        text_file.write(img_info[0]+'\t'+"{:.1f}".format(result)+'\n')