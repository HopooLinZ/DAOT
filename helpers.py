
# import the necessary packages
import imutils
from skimage.transform import pyramid_gaussian
import cv2
 
def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    # print('(H:{},W:{})'.format(image.shape[0], image.shape[1]))
#     yield image


    # compute the new dimensions of the image and resize it
    w = int(image.shape[1] / scale)
    image = imutils.resize(image, width=w)
    print('resize=(H:{},W:{})'.format(image.shape[0], image.shape[1]))
    # if the resized image does not meet the supplied minimum
    # size, then stop constructing the pyramid
    if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
        print("Out of size!")
    else:
        yield image

def pyramid2(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    # while True:
        # compute the new dimensions of the image and resize it
        # w = int(image.shape[1] / scale)
    w = image.shape[1]
    image = imutils.resize(image, width=w)
    # print('(H:{},W:{})'.format(image.shape[0], image.shape[1]))

        # if the resized image does not meet the supplied minimum
        # # size, then stop constructing the pyramid
        # if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
        #     print("Out of size!")
        #     break
        # yield the next image in the pyramid
        # yield image
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def sliding_window_32(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0]-48, stepSize):
        for x in range(0, image.shape[1]-48, stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def sliding_window_64(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0]-32, stepSize):
        for x in range(0, image.shape[1]-32, stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def sliding_window_96(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0]-16, stepSize):
        for x in range(0, image.shape[1]-16, stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
def sliding_window_160(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(16, image.shape[0], stepSize):
        for x in range(16, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
if __name__ == '__main__':
    image = cv2.imread('DB/pictures/part_B_final/train_data/images/IMG_1.jpg')
    # METHOD #2: Resizing + Gaussian smoothing.
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        # show the resized image
        WinName = "Layer {}".format(i + 1)
        cv2.imshow(WinName, resized)
        cv2.waitKey(0)
        resized = resized*255
        cv2.imwrite('./'+WinName+'.jpg',resized)
