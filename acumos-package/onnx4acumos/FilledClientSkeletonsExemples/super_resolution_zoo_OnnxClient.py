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
import super_resolution_zoo_pb2 as pb

# Import your own needed library below
"**************************************"
from numpy import clip
import PIL
# torch imports
import torchvision.transforms as transforms



"**************************************"

# Define your own needed method below
"**************************************"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


"**************************************"

# Preprocessing method define 
def preprocessing(preProcessingInputFileName: str):
    preProcessingInputFile = io.open(preProcessingInputFileName, "rb", buffering = 0)
    preProcessingData = preProcessingInputFile.read()
    preProcessingInput = io.BytesIO(preProcessingData)
    # Import the management of the Onnx data preprocessing below. 
    # The "preProcessingOutput" variable must contain the preprocessing result with type found in run_xx_OnnxModel method signature below 
    "*************************************************************************************************"
    global img_cb, img_cr
    img = PIL.Image.open(preProcessingInput)
    resize = transforms.Resize([224, 224])
    img = resize(img)
    img.show()
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)
    preprocessingResult = to_numpy(img_y)
    "**************************************************************************************************"
    # "PreProcessingOutput" variable affectation with the preprocessing result
    preProcessingOutput  = preprocessingResult
    preProcessingInputFile.close()
    return preProcessingOutput

# Postprocessing method define
def postprocessing(postProcessingInput, outputFileName: str)-> bool:
    output = np.array(postProcessingInput.output).reshape((1,1,672,672))
    # Import the management of the Onnx data postprocessing below. 
    # The "postProcessingInput" variable must contain the data of the Onnx model result with type found in method signature below 
    "*************************************************************************************************"
    global img_cb, img_cr
    img_out_y = output[0] 
    img_out_y = np.array((img_out_y[0] * 255.0))
    img_out_y = clip(img_out_y,0, 255)
    img_out_y = PIL.Image.fromarray(np.uint8(img_out_y), mode='L') 
    final_img = PIL.Image.merge(
        "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, PIL.Image.BICUBIC),
        img_cr.resize(img_out_y.size, PIL.Image.BICUBIC),
      ]).convert("RGB")
    f=io.BytesIO()
    final_img.save(f,format='jpeg')
    imageOutputData = f.getvalue()
    final_img.show() 
    postProcessingResult = imageOutputData 
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


restURL = "http://localhost:3330/model/methods/run_super_resolution_zoo_OnnxModel"
headers = {'Content-Type': 'application/vnd.google.protobuf','accept': 'application/vnd.google.protobuf' }

#Load provided onnx model 
modelFileName = "super_resolution_zoo.onnx"
onnx_model = onnx.load(modelFileName)


def run_super_resolution_zoo_OnnxModel(input: List[np.float32]):
    """ This method run the provided onnx model. """
    inputOnnx = pb.RunSuperResolutionZooOnnxmodelIn()
    oneLine = input.reshape(50176)
    inputOnnx.input.extend(oneLine)
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
   ort_outs = run_super_resolution_zoo_OnnxModel(preprocessingData)
   # postprocessing onnx data, the result is stored in new file 
   postprocessing(ort_outs,outputFileName)
else:
   raise AcumosError("The model {} is not a ONNX Model or is a malformed ONNX model".format(modelFileName))






