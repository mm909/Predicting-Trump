import csv
import time
import numpy as np

class Node(object):
    def __init__(self, data, acc=-1, des='', parent=None):
        self.data = data
        if acc == '':
            self.acc = 0
        else:
            self.acc = float(acc)
        self.children = []
        self.parent = parent
        self.des = des

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.data)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def print(self):
        print(self.data)
        for child in self.children:
            child.print()
            pass
        return

    def add_child(self, data, acc, des):
        child = Node(data, acc, des)
        self.children.append(child)
        return self

    def add_parent_child(self, parent, data, acc, des):
        if parent == '':
            return None
        if self.data == parent:
            return self.add_child(data, acc, des)
        for child in self.children:
            child.add_parent_child(parent, data, acc, des)
            pass
        return None

    def search(self, data):
        if self.data == data:
            return self
        for child in self.children:
            childSearch = child.search(data)
            if childSearch != None:
                return childSearch
            pass
        return None

    def findMax(self):
        if len(self.children) == 0:
            return self
        Acc = []
        for child in self.children:
            Acc.append(child.findMax())
        Acc.append(self)
        maxIndex = 0
        for i, a in enumerate(Acc):
            if Acc[maxIndex].acc <= Acc[i].acc:
                maxIndex = i
            pass
        return Acc[maxIndex]

def pprint_tree(node, file=None, _prefix="", _last=True):
    print(_prefix, "|- " if _last else "|- ", node.data + ": "  + str(node.des) + " - " + str(node.acc), ' (RUNNING)' if (node.acc == 0 and node.data != 'PLANNED') else '', sep="", file=file)
    _prefix += "   " if _last else "|  "
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        _last = i == (child_count - 1)
        pprint_tree(child, file, _prefix, _last)

def getData():
    IDS = []
    PARENTS = []
    DATA = {}
    with open('D:/Predictive-Text/experiments/predictiveTrump/experiments.csv') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                IDS = row[2:]
            if row[1] == 'Parent':
                PARENTS = row[2:]
            if row[0] == 'Test Results':
                testAcc = row[2:]
                DATA['testAcc'] = testAcc
            if row[1] == 'Short':
                ShortDes = row[2:]
                DATA['ShortDes'] = ShortDes
                print(ShortDes)
    return IDS, PARENTS, DATA

def buildForest():
    forest = []

    IDS, PARENTS, DATA = getData()

    # Find roots
    for i, id in enumerate(IDS):
        if(PARENTS[i] == ''):
            root = Node(id, DATA['testAcc'][i], DATA['ShortDes'][i])
            forest.append(root)
        pass

    # Insert parent/child pairs
    for i, id in enumerate(IDS):
        for tree in forest:
            tree.add_parent_child(PARENTS[i], id, DATA['testAcc'][i], DATA['ShortDes'][i])
            pass
        pass
    return forest

forest = buildForest()
for tree in forest:
    pprint_tree(tree)

BestModel = forest[0].findMax()
print("\n Best Model: " + BestModel.data + " - " + str(BestModel.acc))

with open('D:/Predictive-Text/experiments/predictiveTrump/Summary.txt', 'w') as out:
    timestr = time.strftime("%Y%m%d-%H%M%S\n")
    print(timestr, file=out)
    for tree in forest:
        pprint_tree(tree, file=out)
    print("\n Best Model: " + BestModel.data + " - " + str(BestModel.acc), file=out)
