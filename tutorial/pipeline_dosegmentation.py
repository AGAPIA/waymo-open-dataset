# This does segmentation for all the inputs in the TEST_INPUT folder and puts them to TEST_OUTPUT / TEST_OUTPUTCOMP
# The special option SAVE_ONLY_LABELS will be faster to use since it will store only the labels in the output, not the segmented images too !

import pickle
import os
from semanticSegmentation import test as SemanticSegSupport

INPUT_IMAGES_BASEPATH = os.path.join("semanticSegmentation", "TEST_INPUT")
OUTPUT_LABELS_BASEFILEPATH = os.path.join("semanticSegmentation", "TEST_OUTPUT")
OUTPUTCOMP_LABELS_BASEFILEPATH = os.path.join("semanticSegmentation", "TEST_OUTPUTCOMP")
OUTPUT_LABELS_FILENAME = "_labels.pkl"

def getLabelsOutputPathBySegmentName(segmentName):
    return os.path.join(OUTPUT_LABELS_BASEFILEPATH, segmentName, OUTPUT_LABELS_FILENAME)

def save_labels(obj, segmentName):
    outputPath = getLabelsOutputPathBySegmentName(segmentName)
    with open(outputPath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



import re
import pickle
outputDict = {}
numFramesExtracted = 0
def defineLabelsExtractor(numRgbImagesPerFrame):
    global numFramesExtracted

    # Checks if the input file to be segmented is already aggregated
    def isFrameAlreadySegmented(inputFileName, pathToOutputFolder):
        fileName = inputFileName[inputFileName.rfind('/'):]
        res = re.findall(r'\d+', inputFileName)
        frame = int(res[0])
        index = int(res[1])
        pathName = os.path.join(pathToOutputFolder, f"labels_{frame}.pkl")
        return os.path.exists(pathName)

    def dictionaryExtractorFunc(fileName, predObj, pathToOutputFolder):
        # format frame_X_index_Y.jpg
        fileName = fileName[fileName.rfind('/'):]
        global numFramesExtracted
        res = re.findall(r'\d+', fileName)
        frame = int(res[0])
        index = int(res[1])

        if outputDict.get(frame) == None:
            outputDict[frame] = {}

        valuesForFrame = outputDict[frame]
        valuesForFrame[index] = predObj

        if len(valuesForFrame) == numRgbImagesPerFrame:
            pathName = os.path.join(pathToOutputFolder, f"labels_{frame}.pkl")

            with open(pathName, "wb") as f:
                pickle.dump(outputDict[frame], f, protocol=pickle.HIGHEST_PROTOCOL)
            del outputDict[frame]
            numFramesExtracted += 1

    return dictionaryExtractorFunc, isFrameAlreadySegmented

# Given a list of segments folders already processed by extracting stage, run segmentation on all input files, and return what was announced earlier
def runSegmentation(listOfSegments):
    # Define base segmentation modelBaseParams such as model and configs
    allRGBImagesDict = {}
    for segmentName in listOfSegments:
        print(f"================= Segment {segmentName} ===============")

        # Setup segmentation for this segment from dataset
        segmentFiles_InputPath      = os.path.join(INPUT_IMAGES_BASEPATH, segmentName)
        segmentFiles_OutputPath     = os.path.join(OUTPUT_LABELS_BASEFILEPATH, segmentName)
        segmentFiles_OutputCompPath = os.path.join(OUTPUTCOMP_LABELS_BASEFILEPATH, segmentName)

        modelParams = []
        modelParams.extend(["--imgs", segmentFiles_InputPath])
        modelConfigPath = "semanticSegmentation/config/ade20k-resnet50dilated-ppm_deepsup.yaml"
        modelParams.extend(["--cfg", modelConfigPath])
        modelDirPath = "semanticSegmentation/ade20k-resnet50dilated-ppm_deepsup"
        modelCheckPoint = "epoch_20.pth"
        modelParams.extend(["DIR", modelDirPath])
        modelParams.extend(["TEST.checkpoint", modelCheckPoint])
        modelParams.extend(["TEST.result", segmentFiles_OutputPath])
        modelParams.extend(["TEST.resultComp", segmentFiles_OutputCompPath])
        modelParams.extend(["TEST.saveOnlyLabels", '0'])
        modelParams.extend(["DATASET.scaleFactor", '2.0'])

        # Create functors and run extraction
        extractFunctor, decisionFunctor = defineLabelsExtractor(5)  # 5 images per frame expected
        SemanticSegSupport.runTestSample(modelParams, extractFunctor, decisionFunctor)

        if len(outputDict) != 0:
            print("Missing frames ", outputDict.keys())
            assert False, (f"Incomplete sequence ! some frames are still there")
        print(f"Summary of inference: numFrames: {numFramesExtracted}")


SAVE_ONLY_LABELS = False #

if __name__ == "__main__":
    runSegmentation(["10023947602400723454_1120_000_1140_000"])