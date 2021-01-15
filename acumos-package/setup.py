# -*- coding: utf-8 -*-
# ===============LICENSE_START=======================================================
# Acumos Apache-2.0
# ===================================================================================
# Copyright (C) 2020 Orange Intellectual property. All rights reserved.
# ===================================================================================
# This Acumos software file is distributed by Orange
# under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============LICENSE_END=========================================================

from os.path import dirname, abspath, join as path_join,isdir
from setuptools import setup, find_packages
from os import listdir


SETUP_DIR = abspath(dirname(__file__))

# Package consistency check 

print("*******   Package consistency check   *******\n")

print(SETUP_DIR,":")
print("		",listdir(SETUP_DIR),"\n")

if 'onnx4acumos' not in listdir(SETUP_DIR):
    print("Error : no onnx4acumos directory found \n")

if 'docs' not in listdir(SETUP_DIR):
    print("Warning : no docs directory found \n")

for dir in listdir(SETUP_DIR):
    if isdir(path_join(SETUP_DIR, dir)):
       print(path_join(SETUP_DIR, dir)," :")
       print("		",listdir(path_join(SETUP_DIR, dir)),"\n")   

print("*******   Package consistency check end  *******\n")


DOCS_DIR = path_join(SETUP_DIR, 'docs')

with open(path_join(SETUP_DIR, 'onnx4acumos', '_version.py')) as file:
    globals_dict = dict()
    exec(file.read(), globals_dict)
    __version__ = globals_dict['__version__']


def _long_descr():
    '''Yields the content of documentation files for the long description'''
    for file in ('user-guide.rst', 'tutorial/index.rst', 'release-notes.rst', 'developper-guide.rst'):
        doc_path = path_join(DOCS_DIR, file)
        with open(doc_path) as f:
            yield f.read()


setup(
    author='Bruno Lozach',
    author_email='bruno.lozach@orange.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
    ],
    description='Acumos ONNX client library for pushing Onnx models in Acumos',
    entry_points="""
    [console_scripts]
    onnx4acumos=onnx4acumos.acumos_onnx_onboarding:run_app_cli
    """,
    install_requires=['protobuf',
                      'requests',
                      'numpy',
                      'configparser',
                      'dill',
                      'appdirs',
                      'filelock',
                      'grpcio',
                      'zipp',
                      'acumos',
                      'acumos_model_runner',
                      'onnx',
                      'onnxruntime',
                      'typing_inspect'],
    keywords='acumos machine learning model modeling artificial intelligence ml ai onnx',
    license='Apache License 2.0',
    long_description='\n'.join(_long_descr()),
    long_description_content_type="text/x-rst",
    name='onnx4acumos',
    packages=find_packages(),
    package_data={'onnx4acumos': [path_join('Templates', '*.py')]},
    python_requires='>=3.6, <3.9',
    url='https://gerrit.acumos.org/r/gitweb?p=acumos-onnx-client.git',
    version=__version__,
)
