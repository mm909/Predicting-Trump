import re
import os
# from pycontractions import Contractions

# def expandContractions(text):
#     cont = Contractions('../models/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin')
#     print(list(cont.expand_texts(text, precise=True)))
#     return

def removeLinks(text):
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

def removeEmojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def removeEllipsis(text):
    return re.sub('\.\.[\.]*', " ", text)

def removeParens(text):
    return re.sub("[\(\[].*?[\)\]]", "", text)

def removeSpeaker(text):
    list = re.findall('\n[\w* *]*: ', text)
    countList = [[x,list.count(x)] for x in set(list)]
    Remove = [value for value in countList if value[1] > 5]
    print(Remove)
    for name in Remove:
        if name[0][1:] != 'donald trump: ':
            name = name[0].lower()[1:]
            text = re.sub(name + '[\w* ,.*]*\n', '', text)
        else:
            print(name)
        pass
    text = re.sub('[\w* *]*: ', '',text)
    return text

def removeLF(text):
    text = re.sub('\n',' ', text)
    text = re.sub(' [ ]*', ' ', text)
    return text

path = '../data/trump/tweets/clean/cleanTweets.txt'
text = open(path, encoding="utf8").read().lower()
text = removeLinks(text)
text = removeEmojis(text)
# text = removeParens(text)
# text = removeSpeaker(text)
text = removeEllipsis(text)
text = removeLF(text)
# text = expandContractions(text)

tc = open("../data/trump/tweets/clean/cleanTweets2.txt","w", encoding="utf8")
tc.write(text)

# os.remove("../data/trump/speeches/clean/concatSpeech.txt")

# Tweets + Github Speeches + 20 Most recent
# corpus length: 23321467
# unique chars: 67
# num training examples: 7773803
#
#
# Tweets plus Github Speeches
# corpus length: 22782426
# unique chars: 67
# num training examples: 4556478
#
# Tweets
# corpus length: 5204704
# unique chars: 66
# num training examples: 1040933
