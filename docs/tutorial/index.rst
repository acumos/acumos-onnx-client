.. ===============LICENSE_START=======================================================
.. Acumos CC-BY-4.0
.. ===================================================================================
.. Copyright (C) 2020 Orange Intellectual Property. All rights reserved.
.. ===================================================================================
.. This Acumos documentation file is distributed by Orange
.. under the Creative Commons Attribution 4.0 International License (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://creativecommons.org/licenses/by/4.0
..
.. This file is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ===============LICENSE_END=========================================================

.. image:: ../../docs/images/Acumos_logo_white.png


=============================
onnx4acumos Tutorial
=============================

This tutorial provides a brief overview for onnx  models on-boarding on Acumos platform. It's meant to be followed linearly, and some code snippets depend on earlier imports and objects. Full onnx python client examples are available in the **``/acumos-onnx-client/acumos-package/onnx4acumos/FilledClientSkeletonsExemples/``** directory of the `Acumos onnx client repository <https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=tree>`__. 


.. note::  We assume that you have already installed **acumos onnx client** (onnx4acumos) python package.


    .. code:: bash

        pip install onnx4acumos
        






In this tutorial, we use `ONNX GoogLeNet <https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet>`__  (source Caffe BVLC GoogLeNet ==> Caffe2 GoogLeNet ==> ONNX GoogLeNet) as example.


=============================


#.  `Introduction`_
#.  `On-boarding Onnx Model on Acumos Plateform`_
#.  `Filling skeleton client file`_
#.  `Command lines`_
#.  `bvlcGoogleNet_Model example`_
#. `More Examples`_


================================


Introduction
============

    Based on the Acumos python client, 
    we developed a python script able to 
    create the onnx model bundle with all 
    the required files needed by Acumos platform.


===================

    .. image:: ../images/Image1.png
        :width: 100px
        :align: center
        :height: 50px


    For more informations on Acumos see :   `Acumos AI Linux Fondation project  <https://www.acumos.org/>`__ , his  `Acumos AI Wiki <https://wiki.acumos.org/>`_ and his `Documentation <https://docs.acumos.org/en/latest/>`_.


==================


On-boarding Onnx Model on Acumos Plateform
==========================================

.. topic:: On-boarding and Micro-services Architecture 
    
    You can find on-boarding and micro-services architecture of the `Acumos AI Plateform <https://wiki.acumos.org/>`__   below :








====================

    .. image:: ../../docs/images/Image2.png
        :width: 112px
        :align: center
        :height: 63px

====================


    **On-boarding and Micro-services  Flow**
 
    
============================



    .. image:: ../../docs/images/Image3.png
        :width: 112px
        :align: center
        :height: 63px

    The data scientist can onboard his onnx models using the Acumos client library. After that, he can create a  micro-service and deploy it localy or on the cloud.


==============================



    .. image:: ../../docs/images/Image4.png
        :width: 112px
        :align: center
        :height: 63px
     
    At the low level view, the E1 Client Library generate all necessary files in order to on-board the model (metadata, Model binary and Model protobuf definition) .  The onboarding server generate the "Model-solutionID" and provide it to Microservice Generation module in order to stock "Model-Docker"  image in Docker repositiory.  In paralell, the onboarding server save the model  in the Artifact Repository. 


====================



.. topic:: Onboarding Onnx Model in Acumos first step
 
    onnx4acumos is a python program that allows you to on-board an onnx model on an Acumos platform. Based on the Acumos python client, we developed this python script able to create the onnx model bundle with all the required files needed by Acumos platform.


====================

    .. image:: ../../docs/images/Capture1.png
        :width: 112px
        :align: center
        :height: 63px


===========================

    bvlcGoogleNet_model locally dumped with input model file example : 

    
        .. code:: bash

            onnx4acumos  OnnxModels/bvlcGoogleNet_model.onnx -f input/cat.jpg



===========================


    bvlcGoogleNet_model on-boarded model with micro-service activation example : 

    
        .. code:: bash

            onnx4acumos  OnnxModels/bvlcGoogleNet_model.onnx -push -ms 


====================


    This script takes the onnx model as input as well as optional parameters (-f data from the input file for the model input file or -push to download the model on Acumos platform and -ms for the activation of the micro-service). The default parameter (-dump) allows the bundle to be saved locally. In this case, the "ModelName" directory is created and contain all the files needed to test the onnx model locally as you can see below.




====================


.. topic:: "ModelName" directory contents
 
    You can find "ModelName"  directory contents description below :


====================

    .. image:: ../../docs/images/Capture2.png
        :width: 112px
        :align: center
        :height: 63px

====================



    In this directory, you cand find :
        - ModelName_OnnxModelOnboarding.py Python file, 
        - Dumped Model directory, 
        - ModelName_OnnxClient directory
    
    All are described in the picture above.


    In our bvlcGoogleNet_model example, the local server part can be started quite simply as follows:

    .. code:: bash

        acumos_model_runner bvlcGoogleNet_Model/dumpedModel/bvlcGoogleNet_Model/

====================


.. topic:: "ModelName_OnnxClient" directory contents
 
    You can find "ModelName_OnnxClient"  directory contents description below :


====================

    .. image:: ../../docs/images/Capture3.png
        :width: 112px
        :align: center
        :height: 63px


====================





    In this directory, you cand find :
        - Input/Input.data file (the input data file provided as onnx4acumos parameter), 
        - ModelName.onnx file (the onnx model file provided as onnx4acumos parameter),
        - ModelName.proto (protobuf file)
        - ModelName_pb2.py (Python pb2 protobuf file to be imported in the onnx client skeleton)
        - ModelName_OnnxClientSkeleton.py (The python client skeleton file that must be completed in order to communicate with server part)


    The filling of the python client skeleton file is detaled below.

====================




.. topic:: Filling of the python client skeleton file
 
    You can find the python client skeleton file filling desciptions below :


====================

    .. image:: ../../docs/images/Capture4.png
        :width: 112px
        :align: center
        :height: 63px


====================





Filling skeleton client file
=============================
 
    You can find the python client skeleton file that must be completed in order to communicate with server part below :


====================

    .. image:: ../../docs/images/Capture5.png
        :width: 112px
        :align: center
        :height: 63px

The "Onnx model protobuf import" is automatiquely imported (namedModel_Model_pb2.py):


    .. code:: python

        
        # Onnx model protobuf import
        import bvlcGoogleNet_Model_pb2 as pb


All "steps" in order to fill the skeleton of our ONNX GoogLeNet as example are discribed below:

.. note::  For an improvement of the comprehension and  re-reading, it is better to fill added lines between two lines of "********".


====================




First import your own needed libraries:
===============================================

    .. code:: python

        
        # Import your own needed library below
        "**************************************"
        import imageio
        from PIL import Image
        import imagenet1000_clsidx_to_labels as idx_to_labels

        "**************************************"

    


==============================================



Second, define your own needed methods:
==============================================

    .. code:: python

        
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



==============================================



Third, define Preprocessing method:
===============================================

    .. code:: python

        
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


==============================================



Fourth, define Postprocessing method:
===============================================

    .. code:: python

        
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



==============================================



And finally, redefine the REST URL if necessary (by default, localhost on port 3330):
=====================================================================================

    .. code:: python

        
        restURL = "http://localhost:3330/model/methods/run_bvlcGoogleNet_Model_OnnxModel"


.. note::    The final name of the filled skeleton ModelName_OnnxClientSkeleton.py could be  ModelName_OnnxClient.py (the same name without Skeleton, bvlcGoogleNet_Model_OnnxClient.py for our bvlc GoogleNet Model example). 

    More, for our exemple, you need to copy in client directory **imagenet1000_clsidx_to_labels.py** file, the dictionary of index results  to lables translation (example :  **'671'**  for the index result  correspond to  **'off-road motorbike, mountain bike, all-terrain bike, off-roader'**  for label result ).

==============================================



Command lines
===============================================

  You can find all command lines for our bvlcGoogleNet_model example below :


    .. image:: ../../docs/images/Commandes.png
        :width: 112px
        :align: center
        :height: 63px


====================




.. code:: bash

    onnx4acumos OnnxModels/bvlcGoogleNet_Model.onnx -f InputData/car4.jpg
    acumos_model_runner bvlcGoogleNet_Model/dumpedModel/bvlcGoogleNet_Model/
    cd  bvlcGoogleNet_Model/bvlcGoogleNet_Model_OnnxClient
    ls
    python bvlcGoogleNet_Model_OnnxClient.py -f input/car4.jpg



==============================================


bvlcGoogleNet_Model example
===============================================




    .. image:: ../../docs/images/bvlc.png
        :width: 112px
        :align: center
        :height: 63px


====================


In our example above : 

.. code:: bash

    python bvlcGoogleNet_Model_OnnxClient.py -f input/car4.jpg
    python bvlcGoogleNet_Model_OnnxClient.py -f input/BM4.jpeg
    python bvlcGoogleNet_Model_OnnxClient.py -f input/espresso.jpeg
    python bvlcGoogleNet_Model_OnnxClient.py -f input/cat.jpg
    python bvlcGoogleNet_Model_OnnxClient.py -f input/pesan3.jpg






==============================================





More Examples
=============




Below are some additional examples. 




super_resolution_zoo_Model example
==================================




    .. image:: ../../docs/images/superResoZoo.png
        :width: 112px
        :align: center
        :height: 63px


====================




.. code:: bash

    python super_resolution_zoo_OnnxClient.py -f input/cat.jpg








==============================================



Emotion Ferplus Model example
==================================




    .. image:: ../../docs/images/emotionFerPlus.png
        :width: 112px
        :align: center
        :height: 63px


====================




.. code:: bash

    python emotion_ferplus_model_OnnxClient.py -f input/angryMan.png
    python emotion_ferplus_model_OnnxClient.py -f input/sadness.png
    python emotion_ferplus_model_OnnxClient.py -f input/happy.jpg
    python emotion_ferplus_model_OnnxClient.py -f input/joker.jpg

==============================================




This is the End :-)
===================






