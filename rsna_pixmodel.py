import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
#import pylibjpeg
import dicomsdl as dicom
import tensorflow as tf
from timeit import default_timer as timer
import os
import matplotlib.pyplot as plt
import pickle
import concurrent.futures

pathroot = "K:/rsna/"

with tf.device('/device:cpu:0'):
    model = tf.keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(1280,1664,3),
            pooling='max'
            #include_preprocessing=True
        )

def compute_synthesis(mammocase_patient_id, mammocase_image_id, mammocase_view, mammocase_laterality, mammocase_implant,\
    mammocase_cancer,mammocase_difficult_negative_case,mammocase_invasive,mammocase_biopsy,height,width):
    # Interesting fields from the DICOM metadata
    metathick = 0x001811A0    # breast thickness
    metacompr = 0x001811A2    # compression force
    metabitalloc = 0x00280100    # bits allocated
    metabitstored = 0x00280101   # bits stored
    metabithigh = 0x00280102     # highest bit
    # Constants
    metacompr_max = 292.0
    metathick_max = 150.0
    pix_max = 16384.0
    # Catergorial recode
    mammoviewcode = {'CC':[1.0,0.0,0.0,0.0,0.0,0.0],
                    'MLO':[0.0,1.0,0.0,0.0,0.0,0.0],
                    'ML':[0.0,0.0,1.0,0.0,0.0,0.0],
                    'LM':[0.0,0.0,0.0,1.0,0.0,0.0],
                    'LMO':[0.0,0.0,0.0,0.0,1.0,0.0],
                    'AT':[0.0,0.0,0.0,0.0,0.0,1.0],
                    'unknwown':[0.0,0.0,0.0,0.0,0.0,0.0]}
    mammolatcode = {'L':0.0,'R':1.0,'unknown':0.5}

    dcmfilename = os.path.join(pathroot,'train_images/',str(mammocase_patient_id),str(mammocase_image_id)+'.dcm')
    dcmobject=dicom.open(dcmfilename)
    bitdepth = dcmobject[metabithigh]
    if bitdepth is None:
        bitdepth = dcmobject[metabitstored]
        if bitdepth is None:
            bitdepth = dcmobject[metabitalloc]
    if bitdepth is None:
        use_pix_max = pix_max
    else:
        use_pix_max = 2**bitdepth
    mammocompr = dcmobject[metacompr]
    if mammocompr is None:
        mammocompr = metacompr_max/2
    mammothick = dcmobject[metathick]
    if mammothick is None:
        mammothick = metathick_max/2
    if mammocase_view in mammoviewcode.keys():
        mammoview = mammoviewcode[mammocase_view]
    else:
        mammoview = mammoviewcode['unknownn']
    if mammocase_laterality in mammolatcode.keys():
        mammolat = mammolatcode[mammocase_laterality]
    else:
        mammolat = mammolatcode['unknown']
    if mammocase_image_id%10==0:
        thedevice = '/device:gpu:0'
    else:
        thedevice = '/device:cpu:0'
    
    with tf.device(thedevice):
        dcmpixels = tf.constant(dcmobject.pixelData())/use_pix_max
        dcmpixels = tf.stack([dcmpixels,dcmpixels,dcmpixels],axis=2)
        dcmpixels = tf.image.resize_with_pad(dcmpixels,height,width)
        dcmpixels = tf.expand_dims(dcmpixels,axis=0)
        metalist = [mammocompr/metacompr_max,mammothick/metathick_max,mammocase_implant]
        metalist += mammoview
        metalist += [mammolat,mammocase_cancer,mammocase_difficult_negative_case,mammocase_invasive,mammocase_biopsy]
        metatensor = tf.constant(metalist,dtype=tf.float32)
        metatensor = tf.expand_dims(metatensor,axis=0)
        pixoutputtensor = model(dcmpixels,training=False)
        outputtensor = tf.concat([pixoutputtensor,metatensor],axis=1)
    return outputtensor


## Generator definition
class genimg:
    def __init__(self,trainvalds,ratio,height,width,batch_size):
        self.trainvalds = trainvalds
        self.batch_size = batch_size
        size = trainvalds.shape[0]
        self.ncases = size
        self.ntrain = int(size*ratio)
        self.nval = self.ncases - self.ntrain
        self.randomgen=np.random.default_rng()
        iota = np.arange(size)
        self.randomgen.shuffle(iota)
        print(iota[:12])
        self.trainidx = iota[:self.ntrain]
        self.valididx = iota[self.ntrain:]
        self.trainds = trainvalds.iloc[self.trainidx]
        #print(self.trainds)
        self.valds = trainvalds.iloc[self.valididx]
        self.general_width = width
        self.general_height = height

    def make_synthesis_partial(self,ds):
        datastack = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            # Start the load operations and mark each future with its URL
            future_to_mammocase = {executor.submit(compute_synthesis,\
                mammocase.patient_id, mammocase.image_id, mammocase.view, mammocase.laterality, mammocase.implant,\
                mammocase.cancer,mammocase.difficult_negative_case,mammocase.invasive,mammocase.biopsy,\
                self.general_height, self.general_width): mammocase for mammocase in ds.itertuples()}
            k=0
            for future in concurrent.futures.as_completed(future_to_mammocase):
                mammocase = future_to_mammocase[future]
                try:
                    outputtensor = future.result()
                except Exception as exc:
                    print('%r _ %r generated an exception: %s' % (mammocase.patient_id,mammocase.image_id,exc))
                else:
                    datastack.append(outputtensor)
                k+=1
                if k%100==0:
                    print("Processing file #",k)
        with tf.device('/device:cpu:0'):
            datastack = tf.concat(datastack,axis=0)
        return datastack
        
    def make_synthesis(self,synthds_train_filename,synthds_val_filename):
        # Create a folder stacking all the images in smaller size
        #os.mkdir('/kaggle/working/trainimg')
        #trainvalimg_dir = '/kaggle/working/trainvalimg'
        #os.rmdir('/kaggle/working/trainimg')
        #os.mkdir(trainvalimg_dir)
        #for root, dirs, files in os.walk(trainvalimg_dir, topdown=False):
        #    for name in files:
        #        os.remove(os.path.join(root, name))

        # Training data processing
        print("Processing train inputs")
        self.synthds_train = self.make_synthesis_partial(self.trainds)
        print("Writing train synthesis file as ",synthds_train_filename," shape=",self.synthds_train.shape)
        with open(os.path.join(pathroot,synthds_train_filename),'wb') as fp:
            pickle.dump(self.synthds_train,fp)

        # Validation data processing
        print("Processing val inputs")
        self.synthds_val = self.make_synthesis_partial(self.valds)
        print("Writing val synthesis file as ",synthds_val_filename," shape=",self.synthds_val.shape)
        with open(os.path.join(pathroot,synthds_val_filename),'wb') as fp:
            pickle.dump(self.synthds_val,fp)
            
    def reload_synthesis(self,synthds_train_filename,synthds_val_filename):
        print("Reloading train synthesis file")
        with open(os.path.join(pathroot,synthds_train_filename),'rb') as fp:
            self.synthds_train = pickle.load(fp)
        print("Reloading val synthesis file")
        with open(os.path.join(pathroot,synthds_val_filename),'rb') as fp:
            self.synthds_val = pickle.load(fp)


if __name__ == '__main__':

    run_mode=1

    # Check version
    print("Tensorflow version = ",tf.__version__)
    #print("DICOMDSL version = ",dicom.__version__)

    trainmeta = pd.read_csv(os.path.join(pathroot,'train.csv'))
    testmeta = pd.read_csv(os.path.join(pathroot,'test.csv'))

    ## Data preparation
    mygen = genimg(trainmeta,0.95,height=1280,width=1664,batch_size=8)   # was 2560 // 3328

    ## Convert images
    mygen.make_synthesis("synthesis_train_xception.pkl","synthesis_val_xception.pkl")
