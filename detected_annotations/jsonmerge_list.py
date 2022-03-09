# this combine several json files as lists
# input dete*.json
# run at folder with the dete*.json files

import json
import os
from typing import Any, Dict, List
import natsort
from natsort import natsorted
import tqdm
from tqdm import tqdm
import glob
annotations=[]
for annotation_name in natsorted(glob.glob('dete*.json')):
        anns = json.load(open(annotation_name))
        annotations.append(anns)
outf=open('combined_annonations_compositional_list.json', 'w')
json.dump(annotations, outf)
