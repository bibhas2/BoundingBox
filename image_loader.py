import numpy as np
import matplotlib.image as img
import os.path

def load_image_samples(dirName):
    MAX_NUM_SAMPLES = 10000
    imageList = []
    classificationList = []
    imageClassList = ["marker", "not_marker"]

    for imageClassIndex in range(len(imageClassList)):
        for i in range(0, MAX_NUM_SAMPLES):
            #Load the image. The files don't have alpha channel.
            #image will be a 20x20x3 matrix
            imageFile = "{0}/{1}{2}.png".format(dirName, imageClassList[imageClassIndex], i + 1)
            
            if os.path.isfile(imageFile) == False:
                break

            print("Loading:", imageFile)
            image = img.imread(imageFile)

            trainingClassification = np.zeros(len(imageClassList))
            trainingClassification[imageClassIndex] = 1

            #Append this image to the result
            imageList.append(image)
            classificationList.append(trainingClassification)

    return np.array(imageList), np.array(classificationList)
