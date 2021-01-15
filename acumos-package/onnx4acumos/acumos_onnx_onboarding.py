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
Created on Wedn July 29 14:37:11 2020

@author: Bruno Lozach OrangeFrance/TGI/OLS/SOFT_LANNION
"""

# Some standard imports
import os
import io
import numpy as np
import re
import subprocess
import sys
import ast
from sys import argv
import onnx
from onnx import AttributeProto as atpb
from acumos.exc import AcumosError
from collections.abc import Iterable
from os.path import dirname, abspath, join as path_join,isdir
from os import listdir
import configparser



def computeShape(shape:onnx.onnx_ONNX_REL_1_7_ml_pb2.TensorShapeProto):
     outputShape = "("
     reshapeValue = 1
     for dim in shape.dim:
       if dim.dim_value!=0:
          reshapeValue *= dim.dim_value
          outputShape += str(dim.dim_value) +","
       else:
          outputShape += "1,"  
     tmp = ""
     i =0
     for dim in outputShape:
       tmp+= dim
       i+=1
       if i==(len(outputShape)-1):
          break
     outputShape = tmp +")"
     return outputShape,reshapeValue


def computeParam(onnxInput,input_elem_type,param = ""):  
     if isinstance(onnxInput,Iterable):
        for elt  in onnxInput:
           if str(elt.type.tensor_type) is not "":
              param+= "'"+ elt.name+"': '"+ str(input_elem_type[elt.name][elt.type.tensor_type.elem_type]) + "',"
           elif   str(elt.type.sequence_type) is not "":
              param+= "'"+ elt.name+"': 'List["
              res = computeParam(elt.type.sequence_type.elem_type,input_elem_type,param)
              param= res + "]',"
           elif  str(onnxInput.type.map_type) is not "":
              print("elt.type.map_type : ", elt)
              #param += computeParam(elt.type.map_type,input_elem_type,param)
     elif  str(onnxInput.map_type) is not "":
          # print("onnxInput.map_type : ", onnxInput)
          param += "Elt" 
     elif  str(onnxInput.tensor_type) is not "":
          print("onnxInput.tensor_type : ", onnxInput)
     return param

def removeLastCharacter(param:str):
     tmp = ""
     i =0
     for character in param:
       tmp+= character
       i+=1
       if i==(len(param)-1):
          break
     param = tmp +""
     return param


def modifOnnxPB(onnxInput,nb, prefixName = "res"):
# Modify Onnx input/output for test usage only
   oneInput = onnxInput[0]

   oneInput.name =  prefixName + "0"

   for i in range(1,nb):
      onnxInput.extend([oneInput])
      onnxInput[i].name = prefixName + str(i)
      onnxInput[i].type.tensor_type.elem_type = i      
   return onnxInput

def checkConfiguration(configFile:str):
# Checking configuration file concistency 
        
    if not os.path.isfile(configFile):
      print("Configuration file ", configFile," is not found")
      exit()
      
    Config = configparser.ConfigParser()

    Config.read(configFile)

    sections = Config.sections()

    errorMsg = f"\033[31mERROR : Bad configuration in " + configFile + " file :\n\n" + "\033[00m"
    
    Ok = True
    
    if 'certificates' in sections and 'proxy' in sections and  'session' in sections:
       try:
          os.environ['CURL_CA_BUNDLE'] = Config.get('certificates', 'CURL_CA_BUNDLE')
       except:
          errorMsg += "	'CURL_CA_BUNDLE' missing in section [certificates], exemple :\n		[certificates]\n		CURL_CA_BUNDLE: /etc/ssl/certs/ca-certificates.crt\n\n"
          Ok = False
       try:
          os.environ['https_proxy'] = Config.get('proxy', 'https_proxy')
       except:
          errorMsg += "	'https_proxy' missing in section [proxy], exemple :\n		[proxy]\n		https_proxy: socks5h://127.0.0.1:8886/\n		http_proxy: socks5h://127.0.0.1:8886/\n\n"
          Ok = False

       try:
          os.environ['http_proxy'] = Config.get('proxy', 'http_proxy')
       except:
          errorMsg += "	'http_proxy' missing in section: [proxy], exemple :\n		[proxy]\n		https_proxy: socks5h://127.0.0.1:8886/\n		http_proxy: socks5h://127.0.0.1:8886/\n\n"
          Ok = False

       try:
          push_api = Config.get('session', 'push_api')
       except:
         errorMsg += "	'push_api' missing in section: [session], exemple :\n		[session]\n		push_api: https://acumos/onboarding-app/v2/models\n\n"
         Ok = False

    else:
       errorMsg = f"\033[31mSections missing in " + configFile +" Configuration file :\033[00m\n 	All [certificates], [proxy] and [session] sections should be defined and filled (see onnx4acumos documentation)"
       Ok = False
    
    if not Ok:
       print(errorMsg)
       exit()
       
    return Ok 
    

def run_app_cli():

   # Found where is the setup directory 
   SETUP_DIR = abspath(dirname(__file__))

   # Dump by default (not push)
   pushSession = False

   # Push model on on Acumos platform ?
   for arg in argv:
       if re.search("push", arg):
           pushSession = True


   # create_microservice (not by default)?
   createMicroService = False
   for arg in argv:
       if re.search("ms", arg):
           createMicroService = True
   modelPath = ""
   configFile =""
   modelFileName = "Onnx model should be with .onnx extension or "
   for arg in argv:
       if re.search(".onnx", arg) and not re.search("onnx4acumos", arg):
          modelPath = arg
          if pushSession:
             print("Trying to push", modelPath.split(".")[0], "model on Acumos platform")
          else:
             print("Trying to dump", modelPath.split(".")[0], "model in dumpedModel directory")
             
   for arg in argv:
       if re.search(".ini", arg):
          configFile = arg             
                    

   # Bad command line Help 
   if modelPath == "" or configFile== "":
      print("Command line shoud be : onnx4acumos ModelName.onnx configurationFile.ini [-f input.data] [-push [-ms]]")
      exit()

   # Checking configuration File concistency
   if not checkConfiguration(configFile):
      print("Bad configuration file concistency : ",configFile, " (see onnx4acumos documentation to fill it)")     
      exit()
   
   
   modelFileName = modelPath.split("/")[(len(modelPath.split("/")) - 1)]

   #Existence test of the provided Onnx model file
   if not os.path.isfile(modelPath):
      print("Model file ", modelPath," is not found")
      exit()

   try:  
      # Load provided onnx model 
      onnx_model = onnx.load(modelPath)
   except:
      print("File ", modelFileName,"is not a ONNX Model")
      exit()

   # check provided onnx model
   try:
      checkModel = onnx.checker.check_model(onnx_model)
   except:
      #raise AcumosError("The model {} is a malformed ONNX model".format(modelFileName))
      print("The model {} is a malformed ONNX model".format(modelFileName))
      exit()


   # Recovery of the inputs / outputs of the provided Onnx model

   input_all = [node.name for node in onnx_model.graph.input]
   input_initializer =  [node.name for node in onnx_model.graph.initializer]
   net_feed_input = list(set(input_all)  - set(input_initializer))

   onnxInput = []
   for node in onnx_model.graph.input:
      if node.name in net_feed_input:
         onnxInput.extend([node])

   onnxOutput = onnx_model.graph.output


   # inputs / outputs elements type definition (if tensor type -> numpy N-dimensional array (ndarray)-> List[tensor_type.elem_type])
   # Acumos supported types:
   #    <class 'int'>, <class 'float'>, <class 'str'>,  <class 'bool'>
   #    {<enum 'Enum'>,  <class 'bytes'>
   #    <class 'typing.NamedTuple'>}, typing.Dict, typing.List
   #    <class 'numpy.int32'>, <class 'numpy.int64'>, <class 'numpy.float32'>,<class 'numpy.float64'>

   # See file onnx/onnx_ONNX_REL_1_7_ml_pb2.py for protobuf type definitions ( 2 differentes : attribute type or tensor data type)
     # attribute_type = 'UNDEFINED','FLOAT','INT','STR','TENSOR','GRAPH','SPARSE_TENSOR',''FLOATS','INTS','STRS','TENSORS','GRAPHS','SPARSE_TENSORS']
     # tensor_data_type = ['UNDEFINED','FLOAT','UINT8','INT8','UINT16','INT16','INT32','INT64','STRING','BOOL','FLOAT16','DOUBLE','UINT32','UINT64','COMPLEX64','COMPLEX128', 'BFLOAT16']  


   # Input element type definitions (if tensor type -> numpy N-dimensional array (ndarray) -> List[tensor_type.elem_type])

   input_elem_type = {}


   for elt in onnxInput:
      try:
        if str(elt.type.tensor_type) is not "":      
            input_elem_type[elt.name] = ['UNDEFINED','List[np.float32]','List[np.int32]','List[np.int32]','List[np.int32]','List[np.int32]','List[np.int32]','List[np.int64]','str','bool','List[np.float16]','List[np.float64]','List[np.uint32]','List[np.uint64]','List[np.complex64]','List[np.complex128]', 'BFLOAT16']
        else:
            input_elem_type[elt.name] =  ['UNDEFINED','float','int','str','TENSOR','GRAPH','SPARSE_TENSOR','List[float]','List[int]','List[str]','TENSORS','GRAPHS','SPARSE_TENSORS']     
      except:
        input_elem_type[elt.name] =  ['UNDEFINED','float','int','str','TENSOR','GRAPH','SPARSE_TENSOR','List[float]','List[int]','List[str]','TENSORS','GRAPHS','SPARSE_TENSORS'] 



   # Output element type definitions (if tensor type -> numpy N-dimensional array (ndarray) -> List[tensor_type.elem_type])


   output_elem_type = {}


   for elt in onnxOutput:
      try:
        if str(elt.type.tensor_type) is not "":      
            output_elem_type[elt.name] = ['UNDEFINED','List[np.float32]','List[np.int32]','List[np.int32]','List[np.int32]','List[np.int32]','List[np.int32]','List[np.int64]','str','bool','List[np.float16]','List[np.float64]','List[np.uint32]','List[np.uint64]','List[np.complex64]','List[np.complex128]', 'BFLOAT16']
        else:
            output_elem_type[elt.name] =  ['UNDEFINED','float','int','str','TENSOR','GRAPH','SPARSE_TENSOR','List[float]','List[int]','List[str]','TENSORS','GRAPHS','SPARSE_TENSORS'] 
      except:
        output_elem_type[elt.name] =  ['UNDEFINED','float','int','str','TENSOR','GRAPH','SPARSE_TENSOR','List[float]','List[int]','List[str]','TENSORS','GRAPHS','SPARSE_TENSORS'] 


   # Compute Input and Output parameters

   inputParam = ast.literal_eval("{" + removeLastCharacter(computeParam(onnxInput,input_elem_type)) + "}")

   outputParam = ast.literal_eval("{" + removeLastCharacter(computeParam(onnxOutput,output_elem_type)) + "}")

   onnxInputParam =""
   for k in onnxInput:
       onnxInputParam += str(k.name) + ": " + str(inputParam[k.name]) + ","

   onnxInputParam = removeLastCharacter(onnxInputParam)



   # Modify Onnx input/output for test usage only


   ModifOnnxInput = False
   nb = 1

   if ModifOnnxInput:
    modifOnnxPB(onnxInput,nb,"InputSR")

   ModifOnnxOutput = False
   nb = 1

   if ModifOnnxOutput:
       modifOnnxPB(onnxOutput,nb,"OutputSR")


   # Taking into account the specificities of the provided Onnx model 

   # Method signature
   templateMethodSignature = "def runOnnxModel"
   newMethodSignature = "def run_" + modelFileName.split(".")[0] +"_OnnxModel(" + onnxInputParam + ")-> MultipleReturn:\n"

   # Configuration File 
   templateConfigFile = "configFile = " 
   newConfigFile = "configFile = \"" + configFile + "\"\n"
   
   # reshapedInput 
   templateReshapedInput = "    reshapedInput = np.array"
   newReshapedInput = ""
   for elt in onnxInput:
        if re.search("np", inputParam[elt.name]):
             inputShape, inputReshapeValue = computeShape(shape = elt.type.tensor_type.shape)
             newReshapedInput += "    reshaped"+ elt.name + " = np.array" +"("+ elt.name + ", dtype="+ input_elem_type[elt.name][elt.type.tensor_type.elem_type].split("[")[1].split("]")[0] +").reshape(" +  inputShape + ")\n"
        else:
             newReshapedInput += "    reshaped"+ elt.name + " = " + elt.name + "\n"
     


   # ort_Output 
   templateOrt_Output = "    ort_Output = ort_session.run"
   newOrt_Output = templateOrt_Output + "("

   lastElt = (onnxInput[(len(onnxInput)-1)])
   for elt in onnxInput:
       if elt != lastElt:
          newOrt_Output += "reshaped"+ elt.name + ","
       else:
          newOrt_Output += "reshaped"+ elt.name + ")\n"


   # reshapedOutput  
   templateReshapedOutput = "outputData = ort_Output"
   newReshapedOutput = ""


   maxIter = len(onnxOutput) - 1
   i = 0
   newReturn = "    multipleReturn = MultipleReturn("

   for elt in onnxOutput:
   #     if re.search("np", output_elem_type[elt.name][elt.type.tensor_type.elem_type]):
        if re.search("np", outputParam[elt.name]):
           outputShape, outputReshapeValue = computeShape(shape = elt.type.tensor_type.shape)
           newReshapedOutput += "    "+ elt.name + " = ort_Output["+ str(i) + "].reshape(" +  str(outputReshapeValue) + ")\n"
           newReturn +=  elt.name +" = " + elt.name 
        else:
           newReshapedOutput += "    "+ elt.name + " = ort_Output["+ str(i) + "]\n"
           if (outputParam[elt.name] == "List[Elt]"):
              newReturn +=  elt.name +" = convertDictListToNamedTupleList(" + elt.name + ")" 
           else:
              newReturn +=  elt.name +" = " +  elt.name + ""
        if i != maxIter:
            newReturn += ", "
        else:
            newReturn += ")\n"
        i+=1

   #     multipleReturn return
   templateReturn = "    multipleReturn = MultipleReturn"

   # NamedTuple MultiPle Returns Creation
   templateMPRC = "MultipleReturn = create_namedtuple"
   newMPRC = templateMPRC + "(\"MultipleReturn\", [(\'"
   i = 0
   for elt in onnxOutput:
         newMPRC+= elt.name + "\', " + outputParam[elt.name] + ")" 
         if i != maxIter:
            newMPRC+= ", (\'"
         else:
            newMPRC+= "])\n"
         i+=1


   # Provided model file 
   templateModelFile = "modelFileName ="
   newModelFile = "modelFileName = \"" + modelPath +"\"\n"

   # Model definition
   templateModel = "model = Model"
   newModel = "model = Model(run_" + modelFileName.split(".")[0] +"_OnnxModel=run_"+ modelFileName.split(".")[0] +"_OnnxModel)\n"

   # Dump session call
   templateSessionDump = "session.dump"
   newSessionDump = "   session.dump(model, \'" + modelFileName.split(".")[0] + "\',\'" + modelFileName.split(".")[0] + "/dumpedModel\', requirements )\n"
   newSessionDump += "   session.dump_zip(model, \'" + modelFileName.split(".")[0] + "\',\'" + modelFileName.split(".")[0] + "/"+ modelFileName.split(".")[0]+ ".zip\', requirements )\n"


   # Push session call
   templateSessionPush = "session.push"
   newSessionPush = "   session.push(model, \'" + modelFileName.split(".")[0] + "\', requirements=requirements, options=opts)\n"


   templatePushSession = "pushSession ="
   if pushSession:
      newPushSession= "pushSession = True\n"
   else:
      newPushSession= "pushSession = False\n"

   # Create microservice ?
   templateMicroService = "opts = Options"

   if createMicroService:
      newMicroService = "opts = Options(create_microservice=True)\n"
   else:
      newMicroService = "opts = Options(create_microservice=False)\n"

   # Opening onnxModelOnBoarding template file 
   inputFile = open(path_join(SETUP_DIR, 'Templates', 'onnxModelOnBoardingTemplate.py'), "r")

   # model directory creation 
   dirOnnx = modelFileName.split(".")[0] 


   callMkdir = "mkdir " + dirOnnx
   if not os.path.isdir(dirOnnx):
        subprocess.call(callMkdir,shell=True)
        print("Creation of model onnx directory : ", dirOnnx)
   
   # copy configuration file in model onnx directory 
   cpCall = "cp " + configFile + " "+ dirOnnx 
   subprocess.call(cpCall,shell=True)
   # copy onnx model file in model onnx directory    
   cpCall = "cp " + modelPath + " "+ dirOnnx 
   subprocess.call(cpCall,shell=True)

   # Creation of the new onnxModelOnBoarding file with appropriate features 
   outputFileName = dirOnnx +"/"+ modelFileName.split(".")[0] + '_OnnxModelOnBoarding.py'
   outputFile = open(outputFileName, "w")

   for inLine in inputFile:
      if re.search(templateMethodSignature, inLine):
         outputFile.write(newMethodSignature)
      elif re.search(templateConfigFile, inLine):
         outputFile.write(newConfigFile)
      elif re.search(templateOrt_Output, inLine):
         outputFile.write(newOrt_Output)
      elif re.search(templateReshapedOutput, inLine):
         outputFile.write(newReshapedOutput)
      elif re.search(templateReturn, inLine):
         outputFile.write(newReturn)
      elif re.search(templateMPRC, inLine):
         outputFile.write(newMPRC)
      elif re.search(templateModelFile, inLine):
         outputFile.write(newModelFile)
      elif re.search(templateModel, inLine):
         outputFile.write(newModel)
      elif re.search(templateSessionDump, inLine):
         outputFile.write(newSessionDump)
      elif re.search(templateSessionPush, inLine):
         outputFile.write(newSessionPush)
      elif re.search(templateReshapedInput, inLine):
         outputFile.write(newReshapedInput)
      elif re.search(templatePushSession, inLine):
         outputFile.write(newPushSession)
      elif re.search(templateMicroService, inLine):
         outputFile.write(newMicroService)
      else:
         outputFile.write(inLine)

   inputFile.close()
   outputFile.close()


   callPython = sys.executable + " " + outputFileName
   print("Running  \"", callPython, "\"")

   subprocess.call(callPython,shell=True)

   "*** Onnx Client Creation only with Dump session ***" 
   if not os.path.isdir(str(modelFileName.split(".")[0] + "/dumpedModel")):
       exit()


   # Client directory creation 

   dirClient = dirOnnx + "/" + modelFileName.split(".")[0] + "_OnnxClient"


   callMkdir = "mkdir " + dirClient
   if not os.path.isdir(dirClient):
        subprocess.call(callMkdir,shell=True)
        print("Creation of onnx client directory (only with Dump session): ", dirClient)

   callMkdir = "mkdir " + dirClient +"/input"
   if not os.path.isdir(dirClient + "/input"):
        subprocess.call(callMkdir,shell=True)
        print("Creation of onnx client directory (only with Dump session): ", dirClient + "/input")

   callMkdir = "mkdir " + dirClient +"/output"
   if not os.path.isdir(dirClient + "/output"):
        subprocess.call(callMkdir,shell=True)
        print("Creation of onnx client directory (only with Dump session): ", dirClient + "/output")


   # Copy protbuf model to Client directory
   dirModel = dirOnnx + "/dumpedModel/" + modelFileName.split(".")[0] + "/"

   print("Copy protbuf model from ", dirModel, " to ", dirClient)

   modelProto = "model.proto"

   cpCall = "cp " + dirModel + modelProto + "  ./" + dirClient + "/" + modelFileName.split(".")[0] + ".proto"

   subprocess.call(cpCall,shell=True)

   # pb2 python file creation
   protocCall = "protoc " + " ./" + dirClient + "/" + modelFileName.split(".")[0] + ".proto" + "  --python_out=." 

   print("Running ", protocCall)

   subprocess.call(protocCall,shell=True)

   # Copy Onnx model to Client directory
   cpCall = "cp " +  modelPath + "  " + dirClient  

   print("Copy Onnx Model file \"",modelPath, "\" in \"", dirClient,"\" Onnx Client directory")
   subprocess.call(cpCall,shell=True)

   # Copy data input file if asked (-f) to Client directory
   # Data Input File Name 
   inputFileName = ""
   found = False 
   for arg in argv:
       if found:
          inputFileName = arg
          found = False
       if re.search("-f", arg):
          found = True
        

   cpCall = "cp " +  inputFileName + "  " + dirClient + "/input"

   if inputFileName != "":
      #Existence test of the provided data input file
      if not os.path.isfile(inputFileName):
         print("File ", inputFileName,"is not found")
      else:
         cpCall = "cp " +  inputFileName + "  " + dirClient + "/input"
         print("Copy data input file", inputFileName, "to Client directory : ", dirClient+ "/input")
         subprocess.call(cpCall,shell=True)

   # import protobuf _pb2.py file 
   templateImportPb = "import model_pb2 as pb"
   newImportPb = "import " +  modelFileName.split(".")[0] + "_pb2" + " as pb\n"

   # one ligne
   # oneLine = inputData.reshape(50176)
   templateOnLine = "oneLine = inputData"
   newOnLine = "    oneLine = " + onnxInput[0].name + ".reshape(" + str(inputReshapeValue) + ")\n"

   # Input Onnx Extend 
   # inputOnnx.input.extend(oneLine)
   templateExtend = "inputOnnx.input.extend"
   newExtend = "    inputOnnx." + onnxInput[0].name + ".extend(oneLine)\n"


   # reshape output
   # output = np.array(outputOnnx.value).reshape((1,1,672,672))
   templateReshapeOutput = "output = np.array"
   newReshapeOutput = "    output = np.array(outputOnnx.value).reshape(" + outputShape + ")\n"

   # Client onnx model call 
   #   ort_outs = onnxModel
   templateModelCall = "   ort_outs = onnxModel"
   newModelCall = "   ort_outs = run_" + modelFileName.split(".")[0] +"_OnnxModel(preprocessingData)\n" 

   # modify REST URL 
   templateRestURL = "restURL ="
   newRestURL = "restURL = \"http://localhost:3330/model/methods/run_" + modelFileName.split(".")[0] + "_OnnxModel\"\n"

   # preprocessing type 
   templatePreprocessing = "def preprocessing"
   newPreprocessing = "def preprocessing(preProcessingInputFileName: str):\n"

   # postprocessing type 
   templatePostprocessing = "def postprocessing"
   newPostprocessing = "def postprocessing(postProcessingInput, outputFileName: str)-> bool:\n"

   for elt in onnxOutput:
        if re.search("np", output_elem_type[elt.name][elt.type.tensor_type.elem_type]):
           outputShape, outputReshapeValue = computeShape(shape = elt.type.tensor_type.shape)
           newPostprocessing += "    "+ elt.name + " = np.array(postProcessingInput."+ elt.name + ").reshape(" +  str(outputShape) + ")\n"
        else:
           newPostprocessing += "    "+ elt.name + " = postProcessingInput."+ elt.name + "\n"


   # find descriptor Name In and Out in _pb2.py file
   pb2File = dirClient + "/" + modelFileName.split(".")[0] + "_pb2.py"
   inputFile = open(pb2File, "r")

   descriptorName = "name='Run"

   descriptor = {}

   for inLine in inputFile:
      if re.search(descriptorName, inLine):
          if re.search('In', inLine):
             descriptor['in'] = "    inputOnnx = pb." + inLine.split("'")[1] + "()\n"

   inputFile.close()

   templateInputOnnx = "inputOnnx = pb"
   newInputOnnx = descriptor['in']

   newMethodSignature = "def run_" + modelFileName.split(".")[0] +"_OnnxModel(" + onnxInputParam + "):\n"

   # Model File Name modification ("OnnxModels/" directory removing from the path)
   newModelFile = "modelFileName = \"" + modelFileName +"\"\n"

   "**************************************************************************"
   # Opening Onnx client template file 
   inputFile = open(path_join(SETUP_DIR, 'Templates', 'onnxClientTemplate.py'), "r")

   # Creation of the onnx client skeleton file with appropriate features
   print("Creation of the onnx client skeleton file with appropriate features in",dirClient,"directory") 
   outputFileName = dirClient + "/" + modelFileName.split(".")[0] + '_OnnxClientSkeleton.py'
   outputFile = open(outputFileName, "w")

   for inLine in inputFile:
      if re.search(templateMethodSignature, inLine):
         outputFile.write(newMethodSignature)
      elif re.search(templateImportPb, inLine):
         outputFile.write(newImportPb)
      elif re.search(templateInputOnnx, inLine):
         outputFile.write(newInputOnnx)
      elif re.search(templateModel, inLine):
         outputFile.write(newModel)
      elif re.search(templateModelCall, inLine):
         outputFile.write(newModelCall)
      elif re.search(templateRestURL, inLine):
         outputFile.write(newRestURL)
      elif re.search(templateModelFile, inLine):
         outputFile.write(newModelFile)
      elif re.search(templateReshapedInput, inLine):
         outputFile.write(newReshapedInput)
      elif re.search(templateReshapedOutput, inLine):
         outputFile.write(newReshapedOutput)
      elif re.search(templateReshapeOutput, inLine):
         outputFile.write(newReshapeOutput)
      elif re.search(templateOnLine, inLine):
         outputFile.write(newOnLine)
      elif re.search(templateExtend, inLine):
         outputFile.write(newExtend)
      elif re.search(templatePreprocessing, inLine):
         outputFile.write(newPreprocessing)
      elif re.search(templatePostprocessing, inLine):
         outputFile.write(newPostprocessing)
      else:
         outputFile.write(inLine)

   inputFile.close()
   outputFile.close()

 
if __name__ == '__main__':
    # allow direct run of the cli app for debugging
    run_app_cli()


