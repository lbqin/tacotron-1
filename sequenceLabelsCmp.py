#! /usr/bin/env python2.7
#coding=utf-8
#filename: tts_sequenceLabelsCmp.py
import sys
import time
import os
import glob
from os import listdir
from os.path import isfile, join
import numpy

def read_labtest(textlab):
    fid_lab =open(textlab,'rb')
    arrayOLines = fid_lab.readlines()
    numberOfLines = len(arrayOLines)  # get the number of lines in the file
    length = len(arrayOLines[0].split())
    # returnMat = numpy.zeros((numberOfLines, length), dtype=numpy.int)  # prepare matrix to return
    returnMat = numpy.zeros((numberOfLines, length), dtype=numpy.float32)  # prepare matrix to return
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split(' ')
        returnMat[index, :] = listFromLine[:]
        index += 1
    return returnMat

def readCmplength(datadir, file):
    filename = os.path.join(datadir, file)
    with open(filename, 'r') as f:
       metadatas = [line.strip().split('|') for line in f]

    for metadata in metadatas:
        text = metadata[3]
        features = metadata[4]
        labdir = datadir + "/lab0/"
        lab_feature = read_labtest(os.path.join(labdir, features)+'.lab')
        seqens = text.split(' ')
        if (len(seqens)+1!=lab_feature.size//546):
            print(metadata[4],len(seqens)+1,lab_feature.size/546)


if __name__ == "__main__":
    readCmplength('/home/research/data/speech/tacotron/training3/', 'train.txt')
