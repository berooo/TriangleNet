import os

import xml.dom.minidom as minidom
import numpy as np
import cv2 as cv
import openslide

class MaskMaker:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((width, height, 3), np.uint8)

    def line(self, p1, p2, color=(255, 255, 255), width=10):
        cv.line(self.image, p1, p2, (color[2], color[1], color[0]), width, cv.LINE_AA)

    def save(self, filename, format):
        cv.imwrite(filename, self.image)

#dataDir = 'E:\\jiezhichang\\123\\fenge\\jiechangfenge'
dataDir='/mnt/sdb/shibaorong/data/TCGA/cancer_img/colon_jiechang'


for dn in os.listdir(dataDir):
    if dn.find('.') == -1:
        svs = ''
        xml = ''
        for fn in os.listdir(os.path.join(dataDir, dn)):
            if fn[-3:] == 'svs':
                svs = os.path.join(dataDir, dn, fn)
            elif fn[-3:] == 'xml':
                xml = os.path.join(dataDir, dn, fn)
        if svs != '' and xml != '':
            filename = svs[:svs.find('.')]
            slide = openslide.open_slide(svs)
            dom = minidom.parse(xml)
            root = dom.documentElement
            regions = root.getElementsByTagName('Region')
            all_points = []
            for r in regions:
                points = []
                vs = r.getElementsByTagName('Vertex')
                for v in vs:
                    points.append((int(float(v.getAttribute('X'))), int(float(v.getAttribute('Y')))))
                all_points.append(points)

            mm = MaskMaker(slide.dimensions[0], slide.dimensions[1])
            for points in all_points:
                for i in range(len(points) - 1):
                    mm.line(points[i], points[i + 1], width = 100)
            mm.save(f'{filename}_contour.png', 'png')
            print(f'{filename}_contour.png')
            del mm

            mm = MaskMaker(slide.dimensions[0], slide.dimensions[1])
            for points in all_points:
                for i in range(len(points) - 1):
                    mm.line(points[i], points[i + 1], width = 50)
            mm.save(f'{filename}_content.png', 'png')
            print(f'{filename}_content.png')
            del mm




