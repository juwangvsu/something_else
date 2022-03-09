# this combine several jsons as dict, overlapping keys only keep one of them
# input: dete*.json
# run at the folder with the dete*.json files

import json
import jsonmerge
from jsonmerge import merge
import os
from typing import Any, Dict, List
import natsort
from natsort import natsorted
import tqdm
from tqdm import tqdm
import glob
annotations={}
for annotation_name in natsorted(glob.glob('dete*.json')):
        anns = json.load(open(annotation_name))
        annotations=merge(annotations,anns)
outf=open('combined_annonations_compositional.json', 'w')
json.dump(annotations, outf)
