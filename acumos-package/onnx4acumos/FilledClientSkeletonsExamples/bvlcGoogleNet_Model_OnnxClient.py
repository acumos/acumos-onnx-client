#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:55:11 2020

@author: Bruno Lozach OrangeFrance/TGI/OLS/SOFT_LANNION
"""
# Some standard imports
import io
import os
from sys import argv
import re
import requests
import numpy as np

# acumos import
from acumos.modeling import Model, List, Dict, create_namedtuple, create_dataframe

# onnx import
import onnx

# Onnx model protobuf import
import bvlcGoogleNet_Model_pb2 as pb

# Import your own needed library below
"**************************************"
import imageio
from PIL import Image
import imagenet1000_clsidx_to_labels as idx_to_labels

"**************************************"

# Define your own needed method below
"**************************************"

def get_image(path):
    """ Using path to image, return the RGB load image """
    img = imageio.imread(path, pilmode='RGB')  
    image = Image.open(path)
    image = image.resize((448, int(448 * image.height/image.width)))
    image.show()
    return img
    
# Pre-processing function for ImageNet models using numpy
def preprocess(img):   
    """ Preprocessing required on the images for inference with mxnet gluon
    The function takes loaded image and returns processed tensor """

    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

"**************************************"

# Preprocessing method define 
def preprocessing(preProcessingInputFileName: str):
    preProcessingInputFile = io.open(preProcessingInputFileName, "rb", buffering = 0)
    preProcessingData = preProcessingInputFile.read()
    preProcessingInput = io.BytesIO(preProcessingData)
    # Import the management of the Onnx data preprocessing below. 
    # The "preProcessingOutput" variable must contain the preprocessing result with type found in run_xx_OnnxModel method signature below 
    "*************************************************************************************************"
    path = preProcessingInputFileName
    img = get_image(path)
    img = preprocess(img)
    preprocessingResult = img
    "**************************************************************************************************"
    # "PreProcessingOutput" variable affectation with the preprocessing result
    preProcessingOutput  = preprocessingResult
    preProcessingInputFile.close()
    return preProcessingOutput

# Postprocessing method define
def postprocessing(postProcessingInput, outputFileName: str)-> bool:
    prob_1 = np.array(postProcessingInput.prob_1).reshape((1,1000))
    # Import the management of the Onnx data postprocessing below. 
    # The "postProcessingInput" variable must contain the data of the Onnx model result with type found in method signature below 
    "*************************************************************************************************"
    prob = prob_1
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    postProcessingResult = "\nResults : \n	1 : " + str(idx_to_labels.results[a[0]]) + " with " + str(int(prob[a[0]] * 100000)/1000) + " %   \n	2 : " + str(idx_to_labels.results[a[1]]) +  " with " + str(int(prob[a[1]] * 100000)/1000) + " %   \n	3 : " + str(idx_to_labels.results[a[2]]) +  " with " + str(int(prob[a[2]] * 100000)/1000) + " %   \n	4 : " + str(idx_to_labels.results[a[3]]) + " with " + str(int(prob[a[3]] * 100000)/1000) + "%\n"
    print(postProcessingResult)
    "*************************************************************************************************"
    # "postProcessingResult" variable must be affected with the postprocessing result
    # Save the processed data in new file
    if type(postProcessingResult) == str:
        outputFile = io.open(outputFileName.split(".")[0] +".data", "a")
    else:
        outputFile = io.open(outputFileName, "wb", buffering = 0)
    outputFile.write(postProcessingResult)
    outputFile.close()
    return os.path.isfile(outputFileName)


restURL = "http://localhost:3330/model/methods/run_bvlcGoogleNet_Model_OnnxModel"
headers = {'Content-Type': 'application/vnd.google.protobuf','accept': 'application/vnd.google.protobuf' }

#Load provided onnx model 
modelFileName = "bvlcGoogleNet_Model.onnx"
onnx_model = onnx.load(modelFileName)


def run_bvlcGoogleNet_Model_OnnxModel(data_0: List[np.float32]):
    """ This method run the provided onnx model. """
    inputOnnx = pb.RunBvlcgooglenetModelOnnxmodelIn()
    oneLine = data_0.reshape(150528)
    inputOnnx.data_0.extend(oneLine)
    print("*** Call ONNX Runtime Prediction ***")
    result = requests.post(restURL, headers=headers, data=inputOnnx.SerializeToString())
    #print("result = ", result.content)
    outputOnnx = pb.MultipleReturn()
    outputOnnx.ParseFromString(result.content)
    return outputOnnx

# Data Input File Name 
inputFileName =""
found = False 
for arg in argv:
    if found:
       inputFileName = arg
       found = False
    if re.search("-f", arg):
       found = True
        

#Existence test of the provided data input file
if not os.path.isfile(inputFileName):
   print("File ", inputFileName,"is not found")
   exit()

outputFileName = "output/"+ modelFileName.split(".")[0] + '_'  + inputFileName.split(".")[0].split("/")[1] + "_output." + inputFileName.split(".")[1]

# check onnx model
checkModel = onnx.checker.check_model(onnx_model)

if checkModel is None:
   # preprocessing onnx data 
   preprocessingData = preprocessing(inputFileName)
   # Onnx model call 
   ort_outs = run_bvlcGoogleNet_Model_OnnxModel(preprocessingData)
   # postprocessing onnx data, the result is stored in new file 
   postprocessing(ort_outs,outputFileName)
else:
   raise AcumosError("The model {} is not a ONNX Model or is a malformed ONNX model".format(modelFileName))






