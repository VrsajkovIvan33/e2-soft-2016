# NumSumBlackBackground
Sum all the handwritten numbers, which are from the MNIST data set,from an image that has a black background and contains noise. This is done by using a neural network trained with a random selection from the MNIST data set.

There are three proposed solutions:


- `BlackSumRegions.py`
- `BlackSumRegionsFixated.py`
- `Heatmaps.py`

The training of the neural network is done in the `NetworkTrain.py` file. There are four neural network models included, each using a different amount of samples from the MNIST data set for the training.

## BlackSumRegions
Using regions, try to determine the area where the numbers are located in the image. Then pass through the area with a sliding block (window) and find the best match using the predictions of the neural network.
## BlackSumRegionsFixated
Use the same principle as before, find the regions in the image where the numbers are located. This time though use only one block, its location determined by the *centroid* attribute from the region. Only one prediction for each region.
## Heatmaps
Constuct a heatmap for each digit for every image. Using regions, calculate the number of appearances of every digit in each image and sum them.
# Testing
There are two tests included:


- `TestOriginal.py` - calculates the number of matching sums for every image
- `TestModified.py` - calculates the total difference for the correct and the calculated sums for each image and then compares it to the correct total sum

It should be noted that a .txt file containing the correct results is needed for testing.   
# Results
The most accurate and efficient solution is **BlackSumRegionsFixated**, taking the least time to finish and achieving a 16% accuracy using `TestOriginal.py` and a 95.7% accuracy using `TestModified.py` on test images. **BlackSumRegions** achieved a 3% and a 90.5% accuracy on `TestOriginal.py` and `TestModified.py`, respectfully. **Heatmaps** wasn't tested because after 30 minutes the process still was not finished on the test set. Using single images, the process returned less acurate results than the two previous methods.