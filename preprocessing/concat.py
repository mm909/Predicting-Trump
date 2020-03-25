import shutil
import glob

outfilename = '../data/trump/speeches/clean/cleanSpeech.txt'

with open(outfilename, 'wb') as outfile:
    files = glob.glob('../data/trump/speeches/raw/github/*.txt') + glob.glob('../data/trump/speeches/raw/factbase/*.txt')
    for filename in files:
        if filename == outfilename:
            continue
        with open(filename, 'rb') as readfile:
            shutil.copyfileobj(readfile, outfile)
