import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import string
import requests
import xml.etree.cElementTree as ET
from joblib import Parallel, delayed
import multiprocessing
import warnings
import cv2
from Utils import Utils
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


class DataGen:

    def __init__(self, large_img_folder_path, gen_imgs_folder_path, new_folder_name,
                 sample_img_size, overlap_thresh=0.15, num_proposals=40):
        self.l_img_folder_path = large_img_folder_path
        self.new_folder_path = gen_imgs_folder_path + new_folder_name
        self.sample_img_size = sample_img_size
        self.overlap_thresh = overlap_thresh
        self.num_proposals = num_proposals
        self.fonts = ['helveticaneue medium_2.ttf', 'century.ttf','alger.ttf','brlnsr.ttf','brlnsdb.ttf','schlbkb.ttf',
                      'CALIBRIB.ttf','PERB____.TTF', 'BELLB.TTF','GOUDOSB.TTF','BASKVILL.TTF','georgiab.ttf','georgia.ttf',
                      'seguisb.ttf','segoeui.ttf','engr.ttf', 'IMPRISHA.TTF', 'CASTELAR.TTF', 'UBUNTU-M.TTF','UBUNTU-MI.TTF',
                      'UBUNTU-B.TTF', 'ARLRDBD.TTF', 'GILLUBCD.TTF']
        self.check_fonts()
        self.train_or_test = 'Train'
        self.open_dictionary()

    def check_fonts(self):
        '''Check if fonts are available and remove them if not.'''
        availableFonts = []
        for font in self.fonts:
            try:
                ImageFont.truetype('c:/windows/fonts/'+font, 1)
                availableFonts.append(font)
            except FileNotFoundError:
                warnings.warn('Font {0} not found so was removed from the list of avaible fonts.'.format(font))
        if not availableFonts:
            raise Exception('No avaible fonts found.')
        self.fonts = availableFonts

    def open_dictionary(self):
        '''Create a dictionary of words to sample from.'''
        word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = requests.get(word_site)
        WORDS = response.content.splitlines()
        self.WORDS = [word.decode('utf-8') for word in WORDS]
        self.WORDS += [random.choice([' ', '  ']).join(self.WORDS[i]) for i in range(len(self.WORDS), 3)]
        self.WORDS += list(string.ascii_letters + string.digits)
        self.WORDS += [''.join(random.choices(string.digits, k=random.randint(1, 5))) for _ in range(4000)]
        self.WORDS += [''.join(random.choices(string.ascii_letters + string.digits,
                                              k=random.randint(1, 5))) for _ in range(2000)]

    def create_paths(self):
        '''Create file folders if they don't already exist.'''
        for path in ['/Train', '/Test', '/TrainAnn', '/TestAnn', '/TrainMasks', '/TestMasks']:
            for ex in ['/Char', '/Map']:
                if not os.path.exists(self.new_folder_path+path+ex):
                    os.makedirs(self.new_folder_path+path+ex)

    def get_bounding_box(self, mask):
        '''Get bounding box for a given mask.'''
        indexes = np.where(mask > 0)
        mask_ = mask.copy()
        mask_[indexes] = 1

        y, x = indexes
        xmin, xmax = (x.min(), x.max())
        ymin, ymax = (y.min(), y.max())
        # return mask of bounding box
        return mask_[ymin:ymax, xmin:xmax], xmin, ymin, xmax, ymax

    def create_XML(self, wordTupArray, filename):
        '''Create the annotation XML file.'''
        root = ET.Element("annotation")
        ET.SubElement(root, "Filename").text = filename

        for wordTup in wordTupArray:
            self.create_object(root, wordTup)

        tree = ET.ElementTree(root)
        tree.write(self.new_folder_path + '/'+self.train_or_test+'Ann/Char/' + filename.split('.')[0] + '.xml')

    def create_object(self, root, wordTup):
        _, _, rot, text, score, mask, bbox, rbbox = wordTup
        xmin, ymin, xmax, ymax = bbox

        p1, p2, p3, p4 = rbbox

        obj = ET.SubElement(root, "object")

        mask_flat = mask.astype(np.uint8).flatten()
        mask_str = ''.join(map(str, mask_flat))

        ET.SubElement(obj, "Mask").text = mask_str
        ET.SubElement(obj, "Text").text = text
        ET.SubElement(obj, "Rotation").text = str(rot)
        ET.SubElement(obj, "OverlapScore").text = str(score)

        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(xmin)
        ET.SubElement(bb, "ymin").text = str(ymin)
        ET.SubElement(bb, "xmax").text = str(xmax)
        ET.SubElement(bb, "ymax").text = str(ymax)

        rbb = ET.SubElement(obj, "rotbndbox")
        ET.SubElement(rbb, "p1_x").text = str(p1[0])
        ET.SubElement(rbb, "p1_y").text = str(p1[1])
        ET.SubElement(rbb, "p2_x").text = str(p2[0])
        ET.SubElement(rbb, "p2_y").text = str(p2[1])
        ET.SubElement(rbb, "p3_x").text = str(p3[0])
        ET.SubElement(rbb, "p3_y").text = str(p3[1])
        ET.SubElement(rbb, "p4_x").text = str(p4[0])
        ET.SubElement(rbb, "p4_y").text = str(p4[1])

    def propose_words(self, img, generateXML=True):
        '''Randomly generate a number of text proposals.'''
        width, height = img.size
        img = np.array(img)
        masks = []

        for i in range(self.num_proposals):
            # Randomly sample text info
            font = random.choice(self.fonts)
            fontSize = random.randint(20, 60)
            fnt = ImageFont.truetype('c:/windows/fonts/'+font, fontSize)
            word = random.choice(self.WORDS)
            rot = random.uniform(-89, 89)
            # get the width and height of the proposed text
            w, h = ImageDraw.Draw(Image.new('RGB', (0, 0), (0, 0, 0))).textsize(word, fnt)
            # draw the text with the exact width and height
            text_mask = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(text_mask)
            w, h = d.textsize(word, font=fnt)
            d.text((0, 0), word, font=fnt, fill=tuple([0, 0, 0]))
            # ------------------------------------------------------------------------------------

            binary_mask, xmin, ymin, xmax, ymax = self.get_bounding_box(np.array(text_mask)[:,:,3])

            text_mask2 = text_mask.rotate(rot, resample=Image.BILINEAR, expand=True)
            text_mask_temp = np.array(text_mask)
            text_mask2 = np.array(text_mask2)

            rot_point = np.array([text_mask_temp.shape[1],
                                  text_mask_temp.shape[0]])/2
            shift = np.array([(text_mask_temp.shape[1]-text_mask2.shape[1])/2,
                              (text_mask_temp.shape[0]-text_mask2.shape[0])/2], np.int32)

            h, w = text_mask2.shape[:2]
            if w > width or h > height:
                continue
            xcord = random.randint(0, width-w)
            ycord = random.randint(0, height-h)
            indent = np.array([xcord, ycord])
            n_rot = -rot
            p1 = Utils.rotate(rot_point, (xmin-2, ymin-2), n_rot)-shift+indent
            p2 = Utils.rotate(rot_point, (xmax+2, ymin-2), n_rot)-shift+indent
            p3 = Utils.rotate(rot_point, (xmax+2, ymax+2), n_rot)-shift+indent
            p4 = Utils.rotate(rot_point, (xmin-2, ymax+2), n_rot)-shift+indent

            text_mask2 = np.array(text_mask2)
            text_mask2[text_mask2[:, :, 3] >= [50]] = [0, 0, 0, 255]
            text_mask2[text_mask2[:, :, 3] < [50]] = [0, 0, 0, 0]

            temp = img[ycord:ycord+h, xcord:xcord+w, :]
            score = np.sum((text_mask2[:, :, 0] == 0)*(temp[:, :, 0] == 0)) / \
                np.sum((text_mask2[:, :, 0] == 0))

            binary_mask, xmin, ymin, xmax, ymax = self.get_bounding_box(text_mask2[:, :, 3])

            text_mask2 = Image.fromarray(text_mask2)
            masks.append((text_mask2, (xcord, ycord, w, h), rot, word, score, binary_mask,
                         (xmin+xcord, ymin+ycord, xmax+xcord, ymax+ycord,), (p1, p2, p3, p4,)))

        if len(masks) == 0:
            raise ValueError('No generated text could fit on image. Try decreasing fontsize or limiting word length.')

        return masks, img

    def refine_words(self, masks, randomLimit=None):
        '''Refine the sample of proposed words.'''
        masks = sorted(masks, key=lambda x: x[4], reverse=False)
        masks = [masks[0]] + [masks[i] for i in range(1, len(masks)) if masks[i][4] < self.overlap_thresh]
        final_masks = []

        for i, mask in enumerate(masks):
            if i == 0:
                final_masks.append(mask)
                continue
            overlaps = False
            for f_mask in final_masks:
                if Utils.check_intersects(mask[6], f_mask[6]):
                    overlaps = True
            if not overlaps:
                final_masks.append(mask)

        return final_masks

    def generate_sample(self, img):
        '''Generate sample text on a given image.'''
        img_ = img.copy()
        masks, img_ = self.propose_words(img)
        img_ = Image.fromarray(img_)
        masks = self.refine_words(masks)
        seg = np.zeros(img_.size)
        plt.style.use('ggplot')

        for mask, cords, rot, word, score, text_mask, bbox, rbbox in masks:
            img_.paste(mask, cords[:2], mask)
            d = ImageDraw.Draw(img_)
            d.rectangle(bbox, outline='red')
            seg[bbox[1]:bbox[3], bbox[0]:bbox[2]][text_mask > 0] = 1
            p1, p2, p3, p4 = rbbox
            box = np.array([[p1, p2, p3, p4]], np.int32)
            img_ = cv2.polylines(np.array(img_), [box], True, (0, 0, 255),
                                 thickness=1)
            img_ = Image.fromarray(img_)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_)
        plt.show()
        plt.figure(figsize=(10, 10))
        plt.imshow(seg, cmap='gray')
        plt.show()

        return img_

    def overlay_characters(self, img, filename, generateXML=True):
        masks, _ = self.propose_words(img)
        masks = self.refine_words(masks)

        if generateXML:
            self.create_XML(masks, filename)
        for mask, cords, _, _, _, _, _, _ in masks:
            img.paste(mask, cords[:2], mask)
        return img

    def generate(self, imagePath, num):
        Image.MAX_IMAGE_PIXELS = None
        mapC = 0
        charC = 0
        img = Image.open(imagePath)
        w, h = img.size

        # generate no text images
        for _ in range(self.samples_per_image):
            break
            filename = str(num)+'_'+str(mapC).zfill(8) + '.png'
            savePath = self.new_folder_path + '/{0}/Map/{1}'.format(self.train_or_test, filename)
            img.crop(Utlis.get_random_crop_box(w, h, self.sample_img_size)).convert('1').save(savePath)
            mapC += 1

        # generate images with text overlay
        for _ in range(self.samples_per_image):
            filename = str(num)+'_'+str(charC).zfill(8) + '.png'
            savePath = self.new_folder_path + '/{0}/Char/{1}'.format(self.train_or_test, filename)
            newImg = img.crop(Utils.get_random_crop_box(w, h, self.sample_img_size))
            self.overlay_characters(newImg, filename).convert('1').save(savePath)
            charC += 1

    def generate_dataset(self, numImages, train_or_test='Train'):
        self.train_or_test = 'Train'
        if train_or_test.lower() == 'test':
            self.train_or_test = 'Test'

        imagePaths = glob.glob(self.l_img_folder_path+r'/BW*')
        numPaths = len(imagePaths)
        self.samples_per_image = numImages//numPaths
        for num in range(len(imagePaths)):
            self.generate(imagePaths[num], num)

    def parallel_generate_dataset(self, numImages, train_or_test='Train', num_cores=0):
        self.train_or_test = 'Train'
        if train_or_test.lower() == 'test':
            self.train_or_test = 'Test'

        imagePaths = glob.glob(self.l_img_folder_path+r'/BW*')

        if num_cores == 0:
            num_cores = multiprocessing.cpu_count()
        # look at the same image on different cores without introducing image bias
        print(len(imagePaths))
        if num_cores > 2 * len(imagePaths):
            wastedCores = num_cores // len(imagePaths)
            imagePaths = imagePaths * wastedCores
        self.samples_per_image = numImages // len(imagePaths)

        Parallel(n_jobs=num_cores)(delayed(self.generate)(path, n) for n, path in enumerate(imagePaths))
