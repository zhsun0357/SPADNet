import os
import sys
import pdb

datapath = '/media/data1/szh/SPADnet/Rapp_code/nyuv2_eval/hrspad_clean/'
folder_list = os.listdir(datapath)

"""
for folder in folder_list:
    folder = datapath + folder
    
    if os.path.isdir(folder):
        filename_list = os.listdir(folder)

        for filename in filename_list:
            if ".mat" in filename:
                new_filename = filename.replace('_p', '_nl')
                new_filename = new_filename.replace('_hr', '')
                new_filename = new_filename.replace('_clean', '')
                filename = folder + '/' + filename
                new_filename = folder + '/' + new_filename
                command = 'sudo mv {} {}'.format(filename, new_filename)
                os.system(command)
"""

"""
filenum = 0
for folder in folder_list:
    folder = datapath + folder
    
    if os.path.isdir(folder):
        filename_list = os.listdir(folder)

        for filename in filename_list:
            if (".mat" in filename) and ('_nl' in filename):
                filenum += 1

print(filenum)
"""
with open('util/train_clean.txt', 'r') as f:
    trainlist = f.read().splitlines()

with open('util/test_clean.txt', 'r') as f:
    testlist = f.read().splitlines()

with open('util/val_clean.txt', 'r') as f:
    vallist = f.read().splitlines()

alllist = trainlist + testlist + vallist
print(len(alllist))

"""
for file in alllist:
    if not os.path.exists(datapath + file):
        print(file)
        pdb.set_trace()
"""

for file in alllist:
    monofile = file.replace('spad_', '')
    monofile = monofile.replace('.mat', '.png')
    monofile = monofile.replace('nl9', 'pred')
    monopath = '/home/szh/SPADnet_important_exp/mono_nyuv2/'
    monofile = monopath + monofile

    if not os.path.exists(monofile):
        print(file)
        pdb.set_trace()
