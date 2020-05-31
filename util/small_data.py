import os
import sys
import re

listfile = 'val_clean.txt'
listfile_small = 'val_clean_small.txt'

with open(listfile, 'r') as f:
    filename_list = f.read().splitlines()

for filename in filename_list:
    if "home_office_0011" in filename:
        with open(listfile_small, 'a+') as f:
            f.write(filename + '\n')


