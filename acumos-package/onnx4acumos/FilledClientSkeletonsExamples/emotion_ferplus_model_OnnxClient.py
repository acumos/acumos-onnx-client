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
import emotion_ferplus_model_pb2 as pb

# Import your own needed library below
"**************************************"
from PIL import Image
import torchvision.transforms as transforms

"**************************************"

# Define your own needed method below
"**************************************"

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax2(x):
    """Compute softmax values for each sets of scores in x."""
    return (np.exp(x) / np.sum(np.exp(x)))

"**************************************"

# Preprocessing method define 
def preprocessing(preProcessingInputFileName: str):
    preProcessingInputFile = io.open(preProcessingInputFileName, "rb", buffering = 0)
    preProcessingData = preProcessingInputFile.read()
    preProcessingInput = io.BytesIO(preProcessingData)
    # Import the management of the Onnx data preprocessing below. 
    # The "preProcessingOutput" variable must contain the preprocessing result with type found in run_xx_OnnxModel method signature below 
    "*************************************************************************************************"
    image_path = preProcessingInputFileName
    input_shape = (1, 1, 64, 64)
    img = Image.open(image_path)
    img = img.resize((64, 64), Image.ANTIALIAS)
    img_data = np.array(img)
    img_data = np.resize(img_data, input_shape)
    preprocessingResult = img_data
    image = Image.open(image_path)
    image = image.resize((448, int(448 * image.height/image.width)))
    image.show()
    "**************************************************************************************************"
    # "PreProcessingOutput" variable affectation with the preprocessing result
    preProcessingOutput  = preprocessingResult
    preProcessingInputFile.close()
    return preProcessingOutput

# Postprocessing method define
def postprocessing(postProcessingInput, outputFileName: str)-> bool:
    Plus692_Output_0 = np.array(postProcessingInput.Plus692_Output_0).reshape((1,8))
    # Import the management of the Onnx data postprocessing below. 
    # The "postProcessingInput" variable must contain the data of the Onnx model result with type found in method signature below 
    "*************************************************************************************************"
    emotion_table = {0:'Neutral', 1:'Happiness', 2:'Surprise', 3:'Sadness', 4:'Anger', 5:'Disgust', 6: 'Fear', 7:'Contempt'}
    scores = Plus692_Output_0 
    prob = softmax(scores)
    prob = np.squeeze(prob)
    classes = np.argsort(prob)[::-1]
    postProcessingResult = "\nEmotion : " + str(emotion_table[classes[1]]) + "\n"
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


restURL = "http://localhost:3330/model/methods/run_emotion_ferplus_model_OnnxModel"
headers = {'Content-Type': 'application/vnd.google.protobuf','accept': 'application/vnd.google.protobuf' }

#Load provided onnx model 
modelFileName = "emotion_ferplus_model.onnx"
onnx_model = onnx.load(modelFileName)


def run_emotion_ferplus_model_OnnxModel(Input3: List[np.float32]):
    """ This method run the provided onnx model. """
    inputOnnx = pb.RunEmotionFerplusModelOnnxmodelIn()
    oneLine = Input3.reshape(4096)
    inputOnnx.Input3.extend(oneLine)
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
   ort_outs = run_emotion_ferplus_model_OnnxModel(preprocessingData)
   # postprocessing onnx data, the result is stored in new file 
   postprocessing(ort_outs,outputFileName)
else:
   raise AcumosError("The model {} is not a ONNX Model or is a malformed ONNX model".format(modelFileName))






