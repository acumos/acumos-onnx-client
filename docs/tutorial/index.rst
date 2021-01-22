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
Full onnx python client examples are available in the **/acumos-onnx-client/acumos-package/onnx4acumos** 
directory of the `Acumos onnx client repository <https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=tree>`__.

We assume that you have already installed ``onnx4acumos`` package.

#.  `On-boarding Onnx Model on Acumos Platform`_
#.  `How to test & run your ONNX model`_
#.  `More Examples`_

On-boarding Onnx Model on Acumos Platform
=========================================

Clone the acumos-onnx-client from gerrit (or from Github)

.. code:: bash

     git clone "ssh://your_gerrit_login@gerrit.acumos.org:29418/acumos-onnx-client" && scp -p -P 29418 your_gerrit_login@gerrit.acumos.org:hooks/commit-msg "acumos-onnx-client/.git/hooks/"
     or
     git clone "ssh://your_gerrit_login@gerrit.acumos.org:29418/acumos-onnx-client"

You will need the two following files for this tutorial :

- The model located at **/acumos-onnx-client/acumos-package/onnx4acumos/OnnxModels/super_resolution_zoo.onnx**
- A configuration file located at **/acumos-onnx-client/acumos-package/onnx4acumos/Templates/onnx4acumos.ini**

For the first version of onnx4acumos client, this configuration file is mandatory whatever the kind of on-boarding you used (CLI or WEB)

onnx4acumos.ini looks like :

.. code:: bash

        [certificates]
        CURL_CA_BUNDLE: /etc/ssl/certs/ca-certificates.crt

        [proxy]
        https_proxy: socks5h://127.0.0.1:8886/
        http_proxy:  socks5h://127.0.0.1:8886/

        [session]
        push_api: https://acumos/onboarding-app/v2/models

**certificates :** location of acumos certificates generated during the installation,
you can also let this parameter empty (CURL_CA_BUNDLE:), in that case you will just
receive a warning.

**proxy** : The proxy you used to reach your acumos platform.

**session** : The on-boarding model push API URL, available in Acumos GUI in the ON-BOARDING MODEL page.

To on-board, by CLI, the super_resolution_zoo model in Acumos platform with micro-service activation, use the following
command line :


.. code:: bash

     onnx4acumos super_resolution_zoo.onnx onnx4acumos.ini -push -ms

In this command line the -push parameter is used to on-board the onnx model directly
in Acumos (CLI on-boarding). You will be prompted to enter your on-boarding token
: onboarding token = "your Acumos login":"authentication token" (example : acumos_user:a2a6a9e8f4gbg3c147eq9g3h).
The "authentication token" can be retrieved in the ACUMOS GUI in your personal settings.
The -ms parameter is used to launch the micro-service creation in Acumos right after the on-boarding.
If -ms is omitted, the model will be on-boarded whithout micro-service generation.
(don't worry, you can create the micro-service later in Acumos))

To on-board by web the super_resolution_zoo model in Acumos platform, follow the next step :

First you have to dump the super_resolution_zoo model locally :

.. code:: bash

     onnx4acumos super_resolution_zoo.onnx onnx4acumos.ini -dump -f input/cat.jpg

Thanks to the command line above a "ModelName" directory ("super_resolution_zoo" directory in our case)
is created and it contains all the files needed to test the onnx model locally, the -f parameter is optional and
is used to add an input data file in the ModelName_OnnxClient folder.

An Acumos model bundle is also created locally and ready to be on-boarded in Acumos manually (Web onboarding).
The default parameter -dump (can be omitted) allows the bundle to be saved locally.

You can find the "ModelName" directory contents description below :

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Capture2.png

In this directory, you cand find :

- ModelName_OnnxModelOnboarding.py : Python file used to onboard a model in Acumos by CLI and/or to dump the model bundle locally.
- Dumped Model directory(model bundle) : Directory that contains all the required files nedded by an Acumos platform.
- Zipped model bundle(ModelName.zip) : zip file (built from Dumped Model directory) ready to be onboarded in Acumos.
- ModelName_OnnxClient directory : Directory that contains all the necessary files to create a client/server able to test & run your model.

Then The last thing to do is to drag and drop the Zipped model bundle in the "ON-BOARDING BY WEB" page of Acumos or use the browse function to on-board your
model.

How to test & run your ONNX model
=================================

This on-boarding client can also be used to test and run your onnx model, regardless of whether you want to on-board it or not in Acumos.
You have to follow the two main steps, first Launch the model runner server and then fill the skeleton client file to create the onnx client.

We assume that:

- You have installed `acumos_model_runner <https://pypi.org/project/acumos-model-runner/>`__ package.
- You have dumped the model bundle locally as explained above.

We use a client-server architecture to test and run onnx models, first you have to launch your model runner locally to create the server,
then you have to use a python sript as an onnx client to interact with the server.

Launch model runner server
==========================

The local server part can be started quite simply as follows :

.. code:: bash

    acumos_model_runner super_resolution_zoo/dumpedModel/super_resolution_zoo

The acumos model runner will also create a swagger interface available at localhost:3330.

Fill skeleton client file to create the ONNX client
===================================================

You can find the python client skeleton file desciptions below :

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Capture4.png

This python client skeleton file is available in the following folder  **super_resolution_zoo/super_resolution_zoo_OnnxClient**

All steps, in order to fill this python client skeleton, are described below. You must filled the part between two lines of "***********"
You just have to copy/paste the following code snipsets below in the right place of your skeleton file.

First import your own needed libraries:
=======================================

.. code:: python

        # Import your own needed library below
        "**************************************"
        from numpy import clip
        import PIL
        # torch imports
        import torchvision.transforms as transforms
        "**************************************"

Second, define your own needed methods:
=======================================

.. code:: python

        # Define your own needed method below
        "**************************************"
        def to_numpy(tensor):
             return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        "**************************************"

Third, define Preprocessing method:
===================================

.. code:: python

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

Fourth, define Postprocessing method:
=====================================

.. code:: python

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

And finally :
=============

Redefine the REST URL if necessary (by default, localhost on port 3330):


.. code:: python

        restURL = "http://localhost:3330/model/methods/run_super_resolution_zoo_OnnxModel"

The final name of the filled skeleton ModelName_OnnxClientSkeleton.py could be  ModelName_OnnxClient.py
(the same name without Skeleton, super_resolution_zoo_OnnxClient.py for our example).

The filled python client skeleton file can be retrieved in the acumos-onnx-client folder :
acumos-onnx-client/acumos-package/onnx4acumos/FilledClientSkeletonsExemples/super_resolution_zoo_OnnxClient.py.

Remark : To test super_resolution_zoo you must have a server X running on your local system.

Command lines
=============

You can find all command lines to test and/ run onnx model super_resolution_zoo below :

.. code:: bash

    onnx4acumos super_resolution_zoo.onnx onnx4acumos.ini -f InputData/cat.jpg
    acumos_model_runner super_resolution_zoo/dumpedModel/super_resolution_zoo/ ## Launch the model runner server
    python super_resolution_zoo_OnnxClient.py -f input/cat.jpg ## Launch client and send input data

super_resolution_zoo_Model example
==================================

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/superResoZoo.png

More Examples
=============

Below are some additional examples.
Post and Pre-processing methods are available in the Github folder : `onnx/models <https://github.com/onnx/models>`__

GoogLeNet
=========

You can find all command lines for GoogleNetexample below :

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/Commandes.png

.. code:: bash

    onnx4acumos OnnxModels/GoogleNet.onnx onnx4acumos.ini -f InputData/car4.jpg
    acumos_model_runner GoogLeNet/dumpedModel/GoogleNet/ ## Lanch the model runner server
    cd  GoogLeNet/GoogLeNet_OnnxClient
    python GoogLeNet_OnnxClient.py -f input/car4.jpg ## Launch client and send input data

.. image:: https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git;a=blob_plain;f=docs/images/bvlc.png

In our example above :

.. code:: bash

    python GoogLeNet_OnnxClient.py -f input/car4.jpg
    python GoogLeNet_OnnxClient.py -f input/BM4.jpeg
    python GoogLeNet_OnnxClient.py -f input/espresso.jpeg
    python GoogLeNet_OnnxClient.py -f input/cat.jpg
    python GoogLeNet_OnnxClient.py -f input/pesan3.jpg

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
