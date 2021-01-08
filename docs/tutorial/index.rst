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

====================
onnx4acumos Tutorial
====================

This tutorial explains how to on-board an onnx model in an Acumos platform with microservice creation.
It's meant to be followed linearly, and some code snippets depend on earlier imports and objects.
Full onnx python client examples are available in the
**``/acumos-onnx-client/acumos-package/onnx4acumos/FilledClientSkeletonsExemples/``**
directory of the `Acumos onnx client repository
<https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=tree>`__.

We assume that you have already installed ``onnx4acumos`` package.

In this tutorial, we use `ONNX GoogLeNet <https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet>`__
(source Caffe BVLC GoogLeNet ==> Caffe2 GoogLeNet ==> ONNX GoogLeNet) as example.

Download the file located in the googlenet model page : `here <https://github.com/onnx/models/blob/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx>`__
Then rename it "GoogLeNet.onnx".

#.  `On-boarding Onnx Model on Acumos Platform`_
#.  `How to test & run your ONNX model`_
#.  `More Examples`_

On-boarding Onnx Model on Acumos Platform
=========================================

GoogLeNet model on-boarded in Acumos platform with micro-service activation :

.. code:: bash

     onnx4acumos  OnnxModels/GoogLeNet.onnx -push -ms

In this command line the push parameter is used to on-board the Onnx model directly in Acumos (CLI on-boarding) and the -ms parameter
is used to launch the micro-service creation in Acumos right after the on-boarding.

GoogLeNet model locally dumped with input model file :

.. code:: bash

     onnx4acumos  OnnxModels/GoogLeNet.onnx -f input/cat.jpg

Thanks to the command line above a "ModelName" directory ("GoogLeNet" directory in our case) is created and contain all the files needed to test the onnx model locally, 
the -f parameter is used to add an input data file in the ModelName_OnnxClient folder.

An Acumos model bundle is also created locally and ready to be on-boarded in Acumos manually (Web onboarding). The default parameter
(-dump) allows the bundle to be saved locally.

You can find "ModelName" directory contents description below :

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Capture2.png

In this directory, you cand find :
        - ModelName_OnnxModelOnboarding.py : Python file used to onboard a model in Acumos by CLI and/or to dump the model bundle locally
        - Dumped Model directory(model bundle) : Directory that contains all the required files nedded by an Acumos platform. 
        - Zipped model bundle(ModelName.zip) : zip file (build from Dumped Model directory) ready to be onboarded in Acumos.
        - ModelName_OnnxClient directory : Directory that contains all the necessary files to create a client/server able to test & run your model

How to test & run your ONNX model
=================================

You can find "ModelName_OnnxClient"  directory contents description below :

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Capture3.png

In this directory, you cand find :
        - Input/Input.data file (the input data file provided as onnx4acumos parameter),
        - ModelName.onnx file (the onnx model file provided as onnx4acumos parameter),
        - ModelName.proto (protobuf file)
        - ModelName_pb2.py (Python pb2 protobuf file to be imported in the onnx client skeleton)
        - ModelName_OnnxClientSkeleton.py (The python client skeleton file that must be completed in order to communicate with server part)


If you want to test & run your ONNX model before on-boerded it in Acumos, you have to follow the two main steps.

        1) Launch the model runner server
        2) Fill the skeleton client file to create the ONNX client

Launch model runner server
==========================

In our GoogLeNet model example, the local server part can be started quite simply as follows:

.. code:: bash

    acumos_model_runner GoogLeNet/dumpedModel/GoogLeNet/

The acumos model runner will also create a swagger interface available at localhost:3330.

Fill skeleton client file to create the ONNX client
===================================================

You can find the python client skeleton file filling desciptions below :

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Capture4.png

Here is the python client skeleton file that must be completed in order to communicate with server:

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Capture5.png

The "Onnx model protobuf import" is automatically imported (namedModel_Model_pb2.py) thanks to the first ligne of the
skeleton "import GoogLeNet_pb2 as pb"

All "steps" in order to fill the skeleton of our ONNX GoogLeNet example are discribed below. You must filled the part
between two lines of "***********"

First import your own needed libraries:
=======================================

.. code:: python

        # Import your own needed library below
        "**************************************"
        import imageio
        from PIL import Image
        import imagenet1000_clsidx_to_labels as idx_to_labels

        "**************************************"
   
Second, define your own needed methods:
=======================================

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

Third, define Preprocessing method:
===================================

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

Fourth, define Postprocessing method:
=====================================

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

And finally, redefine the REST URL if necessary (by default, localhost on port 3330):
=====================================================================================

.. code:: python

        restURL = "http://localhost:3330/model/methods/run_GoogLeNet_OnnxModel"

The final name of the filled skeleton ModelName_OnnxClientSkeleton.py could be  ModelName_OnnxClient.py
(the same name without Skeleton, GoogleNet_OnnxClient.py for our GoogleNet Model example).

More, for our exemple, you need to copy in client directory **imagenet1000_clsidx_to_labels.py** file,
the dictionary of index results  to lables translation (example :  **'671'**  for the index result
correspond to  **'off-road motorbike, mountain bike, all-terrain bike, off-roader'**  for label result).

Command lines
=============

You can find all command lines for our bvlcGoogleNet_model example below :

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Commandes.png

.. code:: bash

    onnx4acumos OnnxModels/GoogleNet.onnx -f InputData/car4.jpg 
    acumos_model_runner GoogLeNet/dumpedModel/GoogleNet/ ## Lanch the model runner server
    cd  GoogLeNet/GoogLeNet_OnnxClient
    ls
    python GoogLeNet_OnnxClient.py -f input/car4.jpg ## Launch client and send input data


GoogLeNet example
=================

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/bvlc.png

In our example above :

.. code:: bash

    python GoogLeNet_OnnxClient.py -f input/car4.jpg
    python GoogLeNet_OnnxClient.py -f input/BM4.jpeg
    python GoogLeNet_OnnxClient.py -f input/espresso.jpeg
    python GoogLeNet_OnnxClient.py -f input/cat.jpg
    python GoogLeNet_OnnxClient.py -f input/pesan3.jpg

More Examples
=============

Below are some additional examples.

super_resolution_zoo_Model example
==================================

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/superResoZoo.png

.. code:: bash

    python super_resolution_zoo_OnnxClient.py -f input/cat.jpg

Emotion Ferplus Model example
=============================

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/emotionFerPlus.png

.. code:: bash

    python emotion_ferplus_model_OnnxClient.py -f input/angryMan.png
    python emotion_ferplus_model_OnnxClient.py -f input/sadness.png
    python emotion_ferplus_model_OnnxClient.py -f input/happy.jpg
    python emotion_ferplus_model_OnnxClient.py -f input/joker.jpg

That's all  :-)
===============
