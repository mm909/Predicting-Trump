import os
from os import listdir
from os.path import isfile, join
import time
from shutil import copyfile
import keras


class MLEXPS:

    def __init__(self):
        print('MLEXPS v1')
        self.topic = 'TOPIC'
        self.baseFolder = 'experiments'
        self.exprTimeStamp = 0
        self.exprFilePath = ''
        self.exprWeightPath = ''
        self.copyFileList = []
        self.currModel = None
        self.currArgs = None
        self.models = []
        self.argList = []
        return

    def startExprQ(self):
        if(len(self.models) != len(self.argList)):
            print("Models and Args do not match up.")
            return
        print("Length of queue:", len(self.models))
        for i, expr in enumerate(self.models):
            self.setCurrModel(expr)
            self.setCurrArgs(self.argList[i])
            self.startExpr()
            pass
        return

    def startExpr(self):
        self.currModel.summary()
        self.setupExprDir()
        checkpoint = keras.callbacks.callbacks.ModelCheckpoint(self.exprWeightPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        if 'callbacks' in self.currArgs:
            self.currArgs['callbacks'].append(checkpoint)
        else:
            self.currArgs['callbacks'] = [checkpoint]
        self.currModel.fit(**self.currArgs)
        self.cleanUpWeights()
        return

    def copyFiles(self):
        for file in self.copyFileList:
            copyfile(file, self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp) + '/files' + "/" + file)
        return

    def setupExprDir(self):
        self.exprTimeStamp = time.strftime("%Y%m%d-%H%M%S")

        os.makedirs(self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp), exist_ok=True)
        self.exprFilePath = self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp)
        os.makedirs(self.exprFilePath + '/weights', exist_ok=True)
        os.makedirs(self.exprFilePath + '/logs', exist_ok=True)
        os.makedirs(self.exprFilePath + '/files', exist_ok=True)

        self.exprWeightPath = self.exprFilePath + '/weights' + "/" + "weights-improvement-{epoch:02d}-{val_accuracy:.4f}.hdf5"
        self.copyFiles()

        if(self.currModel):
            with open(self.baseFolder + "/" + self.topic + "/" + str(self.exprTimeStamp) + '/logs' + '/summary.txt', 'w') as file:
                self.currModel.summary(print_fn=lambda x: file.write(x + '\n'))
        return

    def cleanUpWeights(self):
        files = [f for f in listdir(self.exprFilePath + '/weights') if join(self.exprFilePath + '/weights', f)]
        files.pop()
        for file in files:
            if os.path.isfile(self.exprFilePath + '/weights/' + file) and os.path.splitext(file)[1] == '.hdf5':
                os.remove(self.exprFilePath + '/weights/' + file)
        return

    def setModels(self, models):
        self.models = models
        return

    def setCurrModel(self, model):
        self.currModel = model
        return

    def setArgList(self, argList):
        self.argList = argList
        return

    def setCurrArgs(self, currArgs):
        self.currArgs = currArgs
        return

    def setTopic(self, topic):
        self.topic = topic
        return

    def addCopyFile(self, file):
        self.copyFileList.append(file)
        return

    def setCopyFileList(self, files):
        self.copyFileList = files
        return
