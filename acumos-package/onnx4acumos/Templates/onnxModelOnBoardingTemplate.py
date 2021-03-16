# ===============LICENSE_START=======================================================
# Acumos CC-BY-4.0
# ===================================================================================
# Copyright (C) 2020 Orange Intellectual Property. All rights reserved.
# ===================================================================================
# This Acumos documentation file is distributed by Orange
# under the Creative Commons Attribution 4.0 International License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://creativecommons.org/licenses/by/4.0
#
# This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============LICENSE_END=========================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:55:11 2020

@author: Bruno Lozach OrangeFrance/TGI/OLS/SOFT_LANNION
"""
# ===============WARNING==============================================================
# This python program is just an example to show (and help) you how to use onnx model 
# on Acumos platform. It's like a  sand box for the new comers on this topic. 
#
# If you want to push this model on the Acumos platform and you don't have provided 
# the configuration file and/or license file to onnx4acumos python program, you need to set :
# 
# 1) L157 : pushSession= True 
# 2) L160 : configFile= "configurationFileName" (name of provided configuration file, see doc.)
# 3) L174 : opts= Options(create_microservice=False (or True), license="licenseFileName")
#
# Like this example with micro-service creation : 
# 1) L157 : pushSession= True 
# 2) L160 : configFile= "onnx4acumos.ini"
# 3) L174 : opts= Options(create_microservice=True,license="Apache_2.json")
#
# ===============WARNING_END==========================================================

# acumos imports
from acumos.modeling import Model, List, Dict, create_namedtuple, create_dataframe
from acumos.session import AcumosSession, Options, Requirements
from acumos.exc import AcumosError

# Some standard imports
import os
import configparser
import io
import numpy as np

# onnx imports
import onnx
import onnxruntime
import onnxruntime.backend as backend

def checkConfiguration(configFile:str):
# Checking configuration file concistency

    if configFile == "":
      print("Error : configFile variable should be initialized to push on acumos platform")
      exit()

    if not os.path.isfile(configFile):
      print("Configuration file ", configFile," is not found")
      exit()

    global   push_api

    Config = configparser.ConfigParser()

    Config.read(configFile)

    sections = Config.sections()

    errorMsg = f"\033[31mERROR : Bad configuration in " + configFile + " file :\n\n" + "\033[00m"

    Ok = True

    if 'certificates' in sections and 'proxy' in sections and  'session' in sections:
       try:
          os.environ['CURL_CA_BUNDLE'] = Config.get('certificates', 'CURL_CA_BUNDLE')
       except:
          errorMsg += "	'CURL_CA_BUNDLE' missing in section [certificates], example :\n		[certificates]\n		CURL_CA_BUNDLE: /etc/ssl/certs/ca-certificates.crt\n\n"
          Ok = False
       try:
          os.environ['https_proxy'] = Config.get('proxy', 'https_proxy')
       except:
          errorMsg += "	'https_proxy' missing in section [proxy], example :\n		[proxy]\n		https_proxy: socks5h://127.0.0.1:8886/\n		http_proxy: socks5h://127.0.0.1:8886/\n\n"
          Ok = False

       try:
          os.environ['http_proxy'] = Config.get('proxy', 'http_proxy')
       except:
          errorMsg += "	'http_proxy' missing in section: [proxy], example :\n		[proxy]\n		https_proxy: socks5h://127.0.0.1:8886/\n		http_proxy: socks5h://127.0.0.1:8886/\n\n"
          Ok = False

       try:
          push_api = Config.get('session', 'push_api')
       except:
         errorMsg += "	'push_api' missing in section: [session], example :\n		[session]\n		push_api: https://acumos/onboarding-app/v2/models\n\n"
         Ok = False

    else:
       errorMsg = f"\033[31mSections missing in " + configFile +" Configuration file :\033[00m\n 	All [certificates], [proxy] and [session] sections should be defined and filled (see onnx4acumos documentation)"
       Ok = False

    if not Ok:
       print(errorMsg)
       exit()

    return Ok



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
# Warning : if pushSession is True, the configuration file shoud be provided (see documentation)
# and configFile variable below should be initialized in order to push the model on acumos platform
pushSession = False

# configuration file init 
configFile = "onnx4acumos.ini"

if pushSession:
   # Checking configuration file concistency
   checkConfiguration(configFile)
else:
   push_api = ""

req_map = dict(onnx='onnx',onnxruntime='onnxruntime')

requirements = Requirements(req_map=req_map)

session = AcumosSession(push_api=push_api)

opts = Options(create_microservice=False)

model = Model(runOnnxModel=runOnnxModel)

if pushSession:
   # Push onnx model on Acumos plateform
   print("Pushing onnx model on Acumos plateform on : ", push_api)
   session.push(model, 'OnnxModel', requirements=requirements, options=opts)
else:
   # Dump onnx model in dumpedModel directory
   print("Dumping onnx model in dumpedModel directory")
   session.dump(model, 'OnnxModel', '~/Acumos/onnx/onboardOnnxModel/dumpedModel', requirements )





