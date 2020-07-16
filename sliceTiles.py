import os, shutil

#os.environ['PATH'] = "C:\\Zamhown\\Library\\openslide-win64-20171122\\bin" + ";" + os.environ['PATH']

import openslide
from openslide.deepzoom import DeepZoomGenerator
import cv2 as cv
import numpy as np

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

dataDirs = ['/mnt/sdb/shibaorong/data/TCGA/cancer_img/colon_jiechang', '/mnt/sdb/shibaorong/data/TCGA/cancer_img/rectal_zhichang']
contentMaskDirs = ['/mnt/sdb/shibaorong/data/TCGA/contentmask/jiechang', '/mnt/sdb/shibaorong/data/TCGA/contentmask/zhichang']

for index in range(2):
    dirname = dataDirs[index]
    contentMaskDir = contentMaskDirs[index]
    for dn in os.listdir(dirname):
        slide = ''
        svs = ''
        content = ''
        contour = ''
        for fn in os.listdir(os.path.join(dirname, dn)):
            if fn[-3:] == 'svs':
                svs = os.path.join(dirname, dn, fn)
                slide = fn[:23]
            elif fn[-7:-4] == 'our':
                contour = os.path.join(dirname, dn, fn)
        for fn in os.listdir(contentMaskDir):
            if fn[:23] == slide:
                content = os.path.join(contentMaskDir, fn)
        if svs and content and contour:
            print(os.path.join(dirname, dn))

            svs_tiles_dir = os.path.join(dirname, dn, 'svs_tiles')
            content_tiles_dir = os.path.join(dirname, dn, 'content_tiles')
            contour_tiles_dir = os.path.join(dirname, dn, 'contour_tiles')
            
            if os.path.exists(svs_tiles_dir) and os.path.exists(content_tiles_dir) and os.path.exists(contour_tiles_dir) \
                and os.listdir(svs_tiles_dir) and os.listdir(content_tiles_dir) and os.listdir(contour_tiles_dir):
                continue
            if not os.path.exists(svs_tiles_dir):
                os.mkdir(svs_tiles_dir)
            if not os.path.exists(content_tiles_dir):
                os.mkdir(content_tiles_dir)
            if not os.path.exists(contour_tiles_dir):
                os.mkdir(contour_tiles_dir)
            """
            if os.path.exists(svs_tiles_dir):
                shutil.rmtree(svs_tiles_dir)
            if os.path.exists(content_tiles_dir):
                shutil.rmtree(content_tiles_dir)
            if os.path.exists(contour_tiles_dir):
                shutil.rmtree(contour_tiles_dir)
            os.mkdir(svs_tiles_dir)
            os.mkdir(content_tiles_dir)
            os.mkdir(contour_tiles_dir)
            """

            im_content = Image.open(content)
            size_content = im_content.size
            im_content.close()
            right = content[-7:-4] == 'zuo'

            im_contour = Image.open(contour)
            size_contour = im_contour.size
            im_contour.close()
            
            sf = openslide.open_slide(svs)
            data_gen = DeepZoomGenerator(sf, tile_size=2000, overlap=0, limit_bounds=False)
            level = data_gen.level_count - 1
            rows = data_gen.level_tiles[level][1]
            cols = data_gen.level_tiles[level][0]
            if size_content[1] - size_contour[1] > 1:
                width = 2000
                offset = size_contour[0] * 2 - size_content[0]
            else:
                width = 1000
                offset = size_contour[0] - size_content[0]
            j_begin = (offset // width) if right else 0
            j_end = cols if right else min(size_content[0] // width + 1, cols)

            for i in range(rows):
                for j in range(j_begin, j_end):
                    bg = np.zeros((2000, 2000, 3), np.uint8)
                    bg.fill(255)
                    img = np.array(data_gen.get_tile(level, (j, i)))
                    bg[:img.shape[0], :img.shape[1], :] = img[:, :, :3]
                    bg = cv.resize(bg, (1000, 1000))
                    imgpath = os.path.join(svs_tiles_dir, slide + '_' + str(i) + '_' + str(j) + '.png')
                    cv.imwrite(imgpath, bg)
            sf.close()
            del data_gen

            im_contour = Image.open(contour).convert('RGB')
            img_contour = np.array(im_contour)
            for i in range(rows):
                for j in range(j_begin, j_end):
                    bgimg = Image.new('RGB', (1000, 1000), (0, 0, 0))
                    bg = np.array(bgimg)
                    bg[:min(1000, size_contour[1] - i * 1000), :min(1000, size_contour[0] - j * 1000), :3] = \
                        img_contour[i * 1000:min((i + 1) * 1000, size_contour[1]), j * 1000:min((j + 1) * 1000, size_contour[0]), :3]
                    bgimg = Image.fromarray(bg.astype('uint8')).convert('RGB')
                    bgimg.save(os.path.join(contour_tiles_dir, slide + '_contour_' + str(i) + '_' + str(j) + '.png'), 'png')
            im_contour.close()
            del img_contour

            im_content = Image.open(content).convert('RGB')
            img_content = np.array(im_content)
            if right:
                t_offset = offset - j_begin * width
                for i in range(rows):
                    for j in range(j_begin, j_end):
                        bgimg = Image.new('RGB', (width, width), (0, 0, 0))
                        bg = np.array(bgimg)
                        bg[:min(width, size_content[1] - i * width), max(0, offset - j * width):min(width, size_content[0] + offset - j * width), :3] = \
                            img_content[i * width:min((i + 1) * width, size_content[1]), max(0, j * width - offset):min((j + 1) * width - offset, size_content[0]), :3]
                        if width == 2000:
                            bg = cv.resize(bg, (1000, 1000))
                        bgimg = Image.fromarray(bg.astype('uint8')).convert('RGB')
                        bgimg.save(os.path.join(content_tiles_dir, slide + '_content_' + str(i) + '_' + str(j) + '.png'), 'png')
            else:
                for i in range(rows):
                    for j in range(j_begin, j_end):
                        try:

                            tmp_j = j - j_begin
                            bgimg = Image.new('RGB', (width, width), (0, 0, 0))
                            bg = np.array(bgimg)
                            bg[:min(width, size_content[1] - i * width), :min(width, size_content[0] - tmp_j * width), :3] = \
                                img_content[i * width:min((i + 1) * width, size_content[1]), tmp_j * width:min((tmp_j + 1) * width, size_content[0]), :3]
                            if width == 2000:
                                bg = cv.resize(bg, (1000, 1000))
                            bgimg = Image.fromarray(bg.astype('uint8')).convert('RGB')
                            bgimg.save(os.path.join(content_tiles_dir, slide + '_content_' + str(i) + '_' + str(j) + '.png'), 'png')

                        except Exception as e:
                            print(e)
                            continue


            im_content.close()
            del img_content