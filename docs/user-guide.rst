.. ===============LICENSE_START=======================================================
.. Acumos CC-BY-4.0
.. ===================================================================================
.. Copyright (C) 2020 Orange Intellectual Property. All rights reserved.
.. ===================================================================================
.. This Acumos documentation file is distributed by Orange
.. under the Creative Commons Attribution 4.0 International License (the "License")
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

===========
onnx4acumos
===========

|Build Status|

``onnx4acumos`` is a client library that allows modelers to on-board their onnx models
on an Acumos platform and also to test and run their onnx models.

For more informations on Acumos see :
`Acumos AI Linux Fondation project  <https://www.acumos.org/>`__ , his  `Acumos AI Wiki <https://wiki.acumos.org/>`_
and his `Documentation <https://docs.acumos.org/en/latest/>`_.

Based on the ``acumos`` python client, we built ``onnx4acumos`` client able to create the onnx model bundle with all the
required files needed by Acumos platform.
When you used ``onnx4acumos``, you can choose to on-board your onnx model directly in Acumos with or whithout micro-service
creation (CLI on-boarding). Or you can choose to save your Acumos model bundle locally for later manual on-boarding (Web-onboarding).
It that case ``onnx4acumos`` will create a ModelName Directory in which you will find the Acumos model bundle and all the
necessary files to test and run the Acumos onnx model bundle locally.

Micro-service generation in Acumos will transform your onnx model as a serving model, based on docker, ready to be deployed.

Installation
============

The main requirements to install ``onnx4acumos`` is to install first the following dependancies :

onnx, zipp, acumos, acumos-model-runner, numpy, requests, protobuf, dill, appdirs, filelock, typing-inspect, grpcio, onnxruntime

Once it is done, you can install ``onnx4acumos`` with pip:

.. code:: bash

    pip install onnx4acumos

remark : if you used Acumos CLIO you must used python3.6 with acumos 0.8.0 and acumos_model_runner 0.2.3

.. |Build Status| image:: https://jenkins.acumos.org/buildStatus/icon?job=acumos-onnx-client-tox-verify-master
   :target: https://jenkins.acumos.org/job/acumos-onnx-client-tox-verify-master/
