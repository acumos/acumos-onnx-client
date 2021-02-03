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
import Iris_Model_pb2 as pb

# Import your own needed library below
"**************************************"
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import ast


"**************************************"

# Define your own needed method below
"**************************************"

"**************************************"

# Preprocessing method define 
def preprocessing(preProcessingInputFileName: str):
    preProcessingInputFile = io.open(preProcessingInputFileName, "rb", buffering = 0)
    preProcessingData = preProcessingInputFile.read()
    preProcessingInput = io.BytesIO(preProcessingData)
    # Import the management of the Onnx data preprocessing below. 
    # The "preProcessingOutput" variable must contain the preprocessing result with type found in run_xx_OnnxModel method signature below 
    "*************************************************************************************************"
    global indice 
    global preprocessingResult
    preProcessingInputFile = io.open(preProcessingInputFileName, "r")
    preProcessingData = preProcessingInputFile.read()
    preprocessingResult = np.array(ast.literal_eval(preProcessingData))[indice]
    "**************************************************************************************************"
    # "PreProcessingOutput" variable affectation with the preprocessing result
    preProcessingOutput  = preprocessingResult
    preProcessingInputFile.close()
    return preProcessingOutput

# Postprocessing method define
def postprocessing(postProcessingInput, outputFileName: str)-> bool:
    output_label = np.array(postProcessingInput.output_label).reshape((1))
    output_probability = postProcessingInput.output_probability
    # Import the management of the Onnx data postprocessing below. 
    # The "postProcessingInput" variable must contain the data of the Onnx model result with type found in method signature below 
    "*************************************************************************************************"
    proba = "probability("
    for k in range(len(output_probability)):
         if k != (len(output_probability)-1):
            proba+= output_probability[k].key + " : " + str(output_probability[k].value) +", "
         else:
            proba+= output_probability[k].key + " : " + str(output_probability[k].value) +")"

        
    global preprocessingResult
    postProcessingResult = "result("+ str(preprocessingResult) + ") : " + str(output_label[0]) + "   " + proba + "\n"
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


restURL = "http://localhost:3330/model/methods/run_Iris_Model_OnnxModel"
headers = {'Content-Type': 'application/vnd.google.protobuf','accept': 'application/vnd.google.protobuf' }

#Load provided onnx model 
modelFileName = "Iris_Model.onnx"
onnx_model = onnx.load(modelFileName)


def run_Iris_Model_OnnxModel(float_input: List[np.float32]):
    """ This method run the provided onnx model. """
    inputOnnx = pb.RunIrisModelOnnxmodelIn()
    oneLine = float_input.reshape(4)
    inputOnnx.float_input.extend(oneLine)
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
# Client personalization below
global indice
indiceMax = 38
#indiceMax = 1
                       

#Existence test of the provided data input file
if not os.path.isfile(inputFileName):
   print("File ", inputFileName,"is not found")
   exit()

outputFileName = "output/"+ modelFileName.split(".")[0] + '_'  + inputFileName.split(".")[0].split("/")[1] + "_output." + inputFileName.split(".")[1]

# check onnx model
checkModel = onnx.checker.check_model(onnx_model)

if checkModel is None:
  # Client personalization below
   for indice in range(indiceMax):
     # preprocessing onnx data 
     preprocessingData = preprocessing(inputFileName)
     # Onnx model call 
     ort_outs = run_Iris_Model_OnnxModel(preprocessingData)
     # postprocessing onnx data, the result is stored in new file 
     postprocessing(ort_outs,outputFileName)
else:
   raise AcumosError("The model {} is not a ONNX Model or is a malformed ONNX model".format(modelFileName))






