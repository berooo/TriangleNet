import os
import shutil

xmlDir = '/mnt/sdb/shibaorong/data/TCGA/annos/jiechang_xml'
svsDir = '/mnt/sdb/shibaorong/data/TCGA/cancer_svs/colon'
outputDir = '/mnt/sdb/shibaorong/data/TCGA/cancer_img/colon_jiechang'

svs = {}

for root, dirs, files in os.walk(svsDir, topdown=True):
    for fn in files:
        if fn[-3:] == 'svs':
            svs[fn[:23]] = os.path.join(root, fn)

print(svs)

for fn in os.listdir(xmlDir):
    if fn[-3:] == 'xml':
        sample = fn[:16]
        slide = fn[:23]
        if svs.__contains__(slide):
            dirpath = os.path.join(outputDir, sample)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            shutil.copy(svs[slide], os.path.join(dirpath, f'{slide}.svs'))
            shutil.move(os.path.join(xmlDir, fn), os.path.join(dirpath, f'{slide}.xml'))
