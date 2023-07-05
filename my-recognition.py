#!/usr/bin/python3

import jetson.inference
import jetson.utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="resnet18", help="model to use, can be:  googlenet, resnet-18, ect.")
args = parser.parse_args()

img = jetson.utils.loadImage(args.filename)
net = jetson.inference.imageNet(args.network)
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)


print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))