#!/bin/bash

## RUN THIS COMMAND USING bash download.sh

## http://yann.lecun.com/exdb/mnist/

rm -rf data
mkdir data
cd data

## this downloads the zip file that contains the mnist data
echo "Downloading MNIST Data"
curl -L http://cs.brown.edu/courses/csci1470/hw_data/hw1.zip --output hw2_mnist.zip
## this unzips the zip file - you will get a directory named "data" containing the data
unzip hw2_mnist.zip
## this cleans up the zip file, as we will no longer use it
rm hw2_mnist.zip
## Move it around for some file structuring
mv data mnist

echo "downloaded data"
