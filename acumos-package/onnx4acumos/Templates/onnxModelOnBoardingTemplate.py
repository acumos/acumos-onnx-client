#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:55:11 2020

@author: Bruno Lozach OrangeFrance/TGI/OLS/SOFT_LANNION
"""
# acumos imports
from acumos.modeling import Model, List, Dict, create_namedtuple, create_dataframe
from acumos.session import AcumosSession, Options, Requirements
from acumos.exc import AcumosError

# Some standard imports
import os
import io
import numpy as np

# onnx imports
import onnx
import onnxruntime
import onnxruntime.backend as backend


os.environ['CURL_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'
os.environ['https_proxy'] = 'socks5h://127.0.0.1:8886/'
os.environ['http_proxy'] = 'socks5h://127.0.0.1:8886/'


#Load provided onnx model 
modelFileName = "model.onnx"
onnx_model = onnx.load(modelFileName)
Elt = create_namedtuple("Elt", [('key', str),('value', float)])
MultipleReturn = create_namedtuple("MultipleReturn", [('output_label', np.int32), ('output_probability', List[Elt])])


def convertDictListToNamedTupleList(dicList)-> List[Elt]:
    namedTupleList = []
    for eltList in dicList:
        for key in eltList:
          elt = Elt(key = str(key), value = float(eltList[key] + 0.0000000000001))
          namedTupleList.append(elt)
    return namedTupleList



def runOnnxModel(inputData: np.float32, inputData2: int )-> MultipleReturn:
    # compute ONNX Runtime output prediction
    print("*** Compute ONNX Runtime output prediction ***")
    reshapedInput = np.array(inputData, dtype=np.float32).reshape((1,1,224,2245555))
    ort_session = backend.prepare(onnx_model)
    ort_Output = ort_session.run(reshapedInput)
    outputData = ort_Output[0].reshape(451584555)
    multipleReturn = MultipleReturn(output_label = Output_label, output_probability = convertDictListToNamedTupleList(Output_probability))
    return multipleReturn 


# check provided onnx model
checkModel = onnx.checker.check_model(onnx_model)

if checkModel is not None:
   raise AcumosError("The model {} is not a ONNX Model or is a malformed ONNX model".format(modelFileName))
   exit()

# prepare Acumos Dump or Push session

pushSession = False

req_map = dict(onnx='onnx',onnxruntime='onnxruntime')

requirements = Requirements(req_map=req_map)

session = AcumosSession(push_api="https://acumos/onboarding-app/v2/models")

opts = Options(create_microservice=False)

model = Model(runOnnxModel=runOnnxModel)

if pushSession: 
   # Push onnx model on Acumos plateform
   print("Pushing onnx model on Acumos plateform")
   session.push(model, 'OnnxModel', requirements=requirements, options=opts) 
else: 
   # Dump onnx model in dumpedModel directory
   print("Dumping onnx model in dumpedModel directory")
   session.dump(model, 'OnnxModel', '~/Acumos/onnx/onboardOnnxModel/dumpedModel', requirements ) 





