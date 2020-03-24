import shutil
import glob

outfilename = '../data/trump/speeches/text.txt'

with open(outfilename, 'wb') as outfile:
    files = glob.glob('../data/trump/speeches/github/*.txt') + glob.glob('../data/trump/speeches/factbase/*.txt')
    for filename in files:
        if filename == outfilename:
            continue
        with open(filename, 'rb') as readfile:
            shutil.copyfileobj(readfile, outfile)
