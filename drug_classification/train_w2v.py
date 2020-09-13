import xml
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import sys

import gensim
from gensim.models import Word2Vec

from config import *
from utils.util import *

def from_xml(file, idx=-1):
    base_name = os.path.basename(file)
    base_name = base_name.split('.')[0]
    
    with open(file) as f:
        string = f.read()
    
    tree = ET.ElementTree(ET.fromstring(string))
    note = tree.getroot()

    drug_code =[]
    
    # p = note.findall('KIMSPOCParam')[idx]
    for rx in note.find('Diagnosis').find("RxInfo").findall("Rx"):
        drug_code.append(rx.attrib['Code'])
    
    return drug_code

xmls_files = []
normal = "/data/gruads/preprocess/normal"
for i, (p,d,f) in enumerate(os.walk(normal)):
    for file in f:
        xmls_files.append(os.path.join(p,file))
        
corpus = []
for i, xml in enumerate(pbar(xmls_files)):
    corpus.append(from_xml(xml))
    
w2v = Word2Vec(corpus, size=FEATURE_SIZE, window=5
                 , min_count=1, workers=4, sg=1
                ,iter=100)
w2v.save(os.path.join(RESULT,"Word2Vec.model"))
