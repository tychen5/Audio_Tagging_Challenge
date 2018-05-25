from os import listdir
from os.path import isfile, join

mypath = 'model_full'

onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)