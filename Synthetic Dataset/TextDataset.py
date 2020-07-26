import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
Image.MAX_IMAGE_PIXELS = None
import os
import glob
import random
import string
import requests
import xml.etree.cElementTree as ET
from joblib import Parallel, delayed
import multiprocessing
import warnings

class DataGen:

    def __init__(self, imageFolderPath, newFolderPath, newFolderName, subImageSize, overlapThreshold=0.17, numTextProposals=40 ):
        self.imgFoldPath = imageFolderPath
        self.newFoldPath = newFolderPath + newFolderName
        self.subImgSize = subImageSize
        self.threshold = overlapThreshold
        self.numProposals = numTextProposals
        self.fonts = ['helveticaneue medium_2.ttf', 'century.ttf','alger.ttf','brlnsr.ttf','brlnsdb.ttf','schlbkb.ttf',
                      'CALIBRIB.ttf','PERB____.TTF', 'BELLB.TTF','GOUDOSB.TTF','BASKVILL.TTF','georgiab.ttf','georgia.ttf',
                      'seguisb.ttf','segoeui.ttf','engr.ttf']

        self.CheckFonts()
        self.TrainOrTest = 'Train'
        self.OpenDictionary()


    def CheckFonts(self):
        availableFonts = []
        for font in self.fonts:
            try:
                ImageFont.truetype('c:/windows/fonts/'+ font, 1)
                availableFonts.append(font)
            except:
                warnings.warn('Font {0} was not found so it was removed from the list of avaible fonts.'.format(font))
        if not availableFonts:
            raise Exception('No avaible fonts found.')

        self.fonts = availableFonts


    def OpenDictionary(self):
        word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        response = requests.get(word_site)
        WORDS = response.content.splitlines()
        self.WORDS = [word.decode('utf-8') for word in WORDS]
        self.WORDS += list(string.ascii_letters + string.digits)
        self.WORDS += [''.join(random.choices(string.digits, k=random.randint(1,5))) for _ in range(4000)]
        self.WORDS += [''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1,5))) for _ in range(2000)]



    def CreatePaths(self):
        for path in [r'\Train', r'\Test', r'\TrainAnn', r'\TestAnn']:
            for ex in [r'\Char',r'\Map']:
                if not os.path.exists(self.newFoldPath+path+ex):
                    os.makedirs(self.newFoldPath+path+ex)


    def getRandomCropBox(self, W, H):
        left = random.randint(0,W-self.subImgSize[0])
        top = random.randint(0,H-self.subImgSize[1])
        return [left, top, left+self.subImgSize[0], top+self.subImgSize[1]]


    def checkIntersects(self, a, b):
        x1,y1,xx1,yy1 = a
        x2,y2,xx2,yy2 = b

        return not (xx1 < x2 or x1 > xx2 or yy1 < y2 or y1 > yy2)


    def getBoundingBox(self, mask):
        indexes = np.where(mask>0)
        mask_ = mask.copy()
        mask_[indexes] = 1

        y, x =  indexes
        xmin, xmax = (x.min(), x.max())
        ymin, ymax = (y.min(), y.max())

        # return mask of bounding box
        return mask_[ymin:ymax, xmin:xmax], xmin, ymin, xmax, ymax


    def createXML(self, wordTupArray, filename):

        root = ET.Element("annotation")
        ET.SubElement(root, "Filename").text = filename

        for wordTup in wordTupArray:
            self.createObject(root, wordTup)

        tree = ET.ElementTree(root)
        tree.write(self.newFoldPath + '\\'+self.TrainOrTest+'Ann\\Char\\' + filename.split('.')[0] + '.xml')


    def createObject(self, root, wordTup):
        _, _, rot, text, score, mask, bbox = wordTup
        xmin, ymin, xmax, ymax = bbox

        obj = ET.SubElement(root, "object")

        mask_flat = mask.astype(np.uint8).flatten()
        mask_str = ''.join(map(str, mask_flat))

        ET.SubElement(obj, "Mask").text = mask_str
        ET.SubElement(obj, "Text").text = text
        ET.SubElement(obj, "Rotation").text = str(rot)
        ET.SubElement(obj, "OverlapScore").text = str(score)

        bb = ET.SubElement(obj, "bndbox")

        ET.SubElement(bb,"xmin").text = str(xmin)
        ET.SubElement(bb,"ymin").text = str(ymin)
        ET.SubElement(bb,"xmax").text = str(xmax)
        ET.SubElement(bb,"ymax").text = str(ymax)


    def proposeWords(self, img, generateXML=True):

        width, height = img.size
        img = np.array(img)
        masks = []

        for i in range(self.numProposals):
            font = random.choice(self.fonts)
            fontSize = random.randint(35,75) # was 55
            fnt = ImageFont.truetype('c:/windows/fonts/'+ font, fontSize)
            word = random.choice(self.WORDS)
            rot = np.clip(np.random.normal()*40, -80, 80)

            w,h = ImageDraw.Draw(Image.new('RGB', (0,0), (0,0,0))).textsize(word, fnt)
            text_mask = Image.new('RGBA', (w,h), (0,0,0,0))
            d = ImageDraw.Draw(text_mask)


            w, h = d.textsize(word, font=fnt)
            d.text((0, 0), word, font=fnt, fill=tuple([104, 97, 85]))
            text_mask = np.array(text_mask.rotate(rot, expand=True))

            text_mask[text_mask[:,:,3]>=[50]]=[104,97,85,255]
            text_mask[text_mask[:,:,3]<[50]]=[0,0,0,0]

            text_mask = Image.fromarray(text_mask)
            w,h = text_mask.size

            if w > width or h > height:
                continue

            xcord = random.randint(0,width-w)
            ycord = random.randint(0,height-h)

            text_mask = np.array(text_mask)

            temp = img[ycord:ycord+h,xcord:xcord+w,:]
            score = np.sum((text_mask[:,:,0]==104)*(temp[:,:,0]==104))/np.sum((text_mask[:,:,0]==104))

            binary_mask, xmin, ymin, xmax, ymax = self.getBoundingBox(text_mask[:,:,0])

            text_mask = Image.fromarray(text_mask)
            masks.append((text_mask, (xcord, ycord, w, h), rot, word, score, binary_mask,
                          (xmin+xcord,ymin+ycord,xmax+xcord,ymax+ycord,) ))

        if len(masks)==0:
            raise ValueError('No generated text could fit on image. Try decreasing fontsize or limiting word length.')

        return masks


    def refineWords(self, masks, randomLimit=None):
        masks = sorted(masks, key=lambda x: x[4],reverse=False)
        masks = [masks[0]] + [masks[i] for i in range(1,len(masks)) if masks[i][4]<self.threshold]
        final_masks = []

        for i, mask in enumerate(masks):
            #print(mask[4],mask[3])
            if i==0:
                final_masks.append(mask)
                continue
            overlaps = False
            for f_mask in final_masks:
                if self.checkIntersects(mask[6], f_mask[6]):
                    overlaps = True
            if not overlaps:
                final_masks.append(mask)

        return final_masks


    def GenerateSample(self, img):
        img_ = img.copy()
        masks = self.proposeWords(img_)
        masks = self.refineWords(masks)
        seg = np.zeros(img_.size)
        plt.style.use('ggplot')

        for mask, cords, rot, word, score, text_mask, bbox in masks:

            img_.paste(mask,cords[:2],mask)
            d = ImageDraw.Draw(img_)
            d.rectangle(bbox, outline='red' )
            seg[bbox[1]:bbox[3], bbox[0]:bbox[2]][text_mask>0] = 1

        plt.figure(figsize=(10,10))
        plt.imshow(img_)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.imshow(seg,cmap='gray')
        plt.show()


    def OverlayCharacters(self, img, filename, generateXML=True):
        masks = self.proposeWords(img)
        masks = self.refineWords(masks)

        if generateXML:
            self.createXML(masks, filename)

        for mask, cords, _, _, _, _, _ in masks:
            img.paste(mask,cords[:2],mask)

        return img


    def Generate(self, imagePath, num):
        Image.MAX_IMAGE_PIXELS = None
        mapC = 0
        charC = 0

        img = Image.open(imagePath)
        w,h = img.size

        # generate no text images
        for _ in range(self.numPerImage):
            break
            filename = str(num)+'_'+str(mapC).zfill(8) + '.png'
            savePath = self.newFoldPath + r'\{0}\Map\{1}'.format(self.TrainOrTest, filename)
            img.crop(self.getRandomCropBox(w,h)).save(savePath)
            mapC += 1

        # generate images with text overlay
        for _ in range(self.numPerImage):
            filename = str(num)+'_'+str(charC).zfill(8) + '.png'
            savePath = self.newFoldPath + r'\{0}\Char\{1}'.format(self.TrainOrTest, filename)
            newImg = img.crop(self.getRandomCropBox(w,h))
            self.OverlayCharacters(newImg, filename ).save(savePath)
            charC += 1


    def GenerateDataSet(self, numImages, TrainOrTest='Train'):
        assert TrainOrTest=='Train' or TrainOrTest=='Test'
        self.TrainOrTest = TrainOrTest

        imagePaths = glob.glob(self.imgFoldPath+r'\*')
        numPaths = len(imagePaths)
        self.numPerImage = numImages//numPaths

        for num in range(numImages):
            self.Generate(imagePaths[num], num)


    def ParallelGenerateDataSet(self, numImages, TrainOrTest='Train', num_cores=0):
        assert TrainOrTest=='Train' or TrainOrTest=='Test'
        self.TrainOrTest = TrainOrTest

        imagePaths = glob.glob(self.imgFoldPath+r'\*')

        if num_cores==0:
            num_cores = multiprocessing.cpu_count()

        # look at the same image on different cores without introducing image bias
        wastedCores = num_cores // len(imagePaths)
        imagePaths = imagePaths * wastedCores

        self.numPerImage = numImages // len(imagePaths)

        Parallel(n_jobs=num_cores)(delayed(self.Generate)(path,n) for n, path in enumerate(imagePaths))
