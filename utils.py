# Copyright Jonathan Booher 2017. All rights reserverd

from selective_search import SelectiveSearch
import multiresolutionimageinterface as mir
from PIL import Image
import PIL
from matplotlib import pyplot as plt
import time
import copy
import numpy as np
import matplotlib.patches as patches

import csv
import threading
from skimage.filters import threshold_otsu
import cv2
from scipy.stats import threshold
import PIL

# used to get a single image from a given a level using thread i.
def getImage( i , f_name , level=8 , startX=0 , startY=0 , dimX=None , dimY = None , justDs=False , warpDim=None):

    # none checks for needing to initialize
    if getImage.reader[i] == None:
        getImage.reader[i] = mir.MultiResolutionImageReader()

    if getImage.mr_image[i] == None or f_name != getImage.name[i]:

        if getImage.mr_image[i] != None:
            getImage.mr_image[i].close()

        getImage.mr_image[i] = getImage.reader[i].open(f_name )
        getImage.name[i] = f_name
        
                
    assert getImage.mr_image[i] != None , ( 'failed to open' , f_name )
        
    ds = getImage.mr_image[i].getLevelDownsample( level )
    if justDs:
        return ds
    
    if dimX == None:
        dimX,_ = getImage.mr_image[i].getLevelDimensions( level )
    if dimY == None:
        _,dimY = getImage.mr_image[i].getLevelDimensions( level )

    arr = None

    # if we need to warp the image do that else do not
    if warpDim != None:
        arr = np.array( Image.fromarray(getImage.mr_image[i].getUCharPatch(int(startX*ds) , int(startY*ds) , dimX , dimY , level ) ).resize(warpDim) )
    else:
        arr = np.array( Image.fromarray(getImage.mr_image[i].getUCharPatch( int( startX*ds) , int(startY*ds) , dimX , dimY , level ) ) )

   
    return ds , arr

n_threads = 4
getImage.mr_image = [ None ]* n_threads
getImage.reader = [ None ]* n_threads
getImage.name = [ None ]* n_threads

# ********************************************************************

#for debu purposes
def dispImage( img , regions=None , old_ds = None , new_ds = None , delta_level=None):
    fig, ax = plt.subplots(1)
    ax.imshow( img )
    if regions != None:

        import copy
        PIL.Image.fromarray( img ).save ( 'in.jpg' )
        img1 = cv2.imread('in.jpg') 
        for r in regions:
            x = int ( r['rect'][0] * old_ds / new_ds)
            y = int ( r['rect'][1] * old_ds / new_ds) 
            dx= int( r['rect'][2] * new_ds / (4**(4-delta_level)) ) + x
            dy= int( r['rect'][3] * new_ds / (4**(4-delta_level)) ) + y


            cv2.rectangle( img1 , ( int(x),int(y)),(dx,dy),(0,0,255),10)
            
            rect = patches.Rectangle( (int(x) , int(y)) , dx ,
                                    dy  , linewidth=2,edgecolor='r' , facecolor='none')

            ax.add_patch( rect )

        PIL.Image.fromarray( img1 ).save( 'out.jpg' )

    return

def otsu( small_img ):
    #filter black and gray
    img_R = small_img[:,:,0]
    img_G = small_img[:,:,1]
    img_B = small_img[:,:,2]
    black = (img_R < 10 ) & (img_G < 10) & (img_B < 10)
    small_img[:,:,:3][black] = [255,255,255]

    grey = (img_R >= 220 ) & (img_G >= 220) & (img_B >=220)
    small_img[:,:,:3][grey] = [255,255,255]

    gray = cv2.cvtColor( small_img  , cv2.COLOR_RGB2HSV )
    mask = (gray[:,:,0] < 0.06) & (gray[:,:,1] < 0.06) & (gray[:,:,2]< 70)
    small_img[:,:,:3][mask] = [255,255,255]

    return small_img
    #changing to gray scale
    img_gray = cv2.cvtColor( small_img, cv2.COLOR_RGB2GRAY )

    # clahe
    clahe = cv2.createCLAHE()
    img_cl = clahe.apply(img_gray);

    # otsu algorithm
    try:
        thresh = threshold_otsu(img_cl)

        threshold( small_img[:,:,:3] , thresh  , [255,255,255] )
    except:
        pass
    return small_img

# perform the selective search and then upsampling the image
def searchAndUpsample( i , f_name , warpDim , numRegions , initialLevel=8 , delta_level=2 , disp=False , debug=False):

    assert f_name != None , 'you  need to provide a file name'

    
    if searchAndUpsample.s_search == None:
        searchAndUpsample.s_search = SelectiveSearch()

    # get the lowest level
    flg = True
    while flg:
        try:
            old_ds, small_img = getImage( i , f_name , initialLevel )
            flg = False
        except ValueError:
            initialLevel -= 1

    small_img = otsu( small_img )
  
    regions = searchAndUpsample.s_search.search( small_img )
    if debug:
        print ( 'Found    '+str( len(regions) )+ '    regions' )


    level = initialLevel - delta_level

    new_ds = getImage( i , f_name , justDs=True , level=level )
    new_ds = old_ds if new_ds == 0 else new_ds

    upsampled_patches = []
    regions = regions[:numRegions]

    if disp:
        _, lg_img = getImage( i , f_name , level=level )
        lg_img = otsu( lg_img )
        dispImage( lg_img , regions , old_ds , new_ds , delta_level)

    
    # get the patches from the upsampled image
    for r in regions:
        x = r['rect'][0] * old_ds / new_ds
        y = r['rect'][1] * old_ds / new_ds
        dx= int( r['rect'][2] * new_ds / (4**(4-delta_level)) )
        dy= int( r['rect'][3] * new_ds / (4**(4-delta_level)) )

        new_ds, img = getImage( i , f_name , level=level , startX=x , startY=y , dimX=dx , dimY=dy , warpDim=warpDim)
        upsampled_patches.append( otsu( img ) )
        
        if disp:
            dispImage( img )

    return np.array( upsampled_patches )

searchAndUpsample.s_search = None


# used to get an image iteratively from f_name going down.
# NOTE: training examples andt esting examplesa re assumed to be in
#     diferent csv files.
#     format of csv files is FILE_NAME , Class , Class , ...
#     where filename is the name of the tif file
def getInput( f_name , i , numRegions , warpDim , quiet=True):

    with open( f_name  , 'r' ) as infile:
        r = csv.reader( infile , delimiter=',' )
        r = list(r )
        while True:
            cnt = 0
            high = len(r)

            # this way we dont try to open the same file on differnt threads
            lock = threading.Lock()
            lock.acquire()
            if getInput.ind >= high:
                getInput.ind = 0
            ind = getInput.ind
            getInput.ind += 1
            lock.release()

            name = r[ind][0]
            patient = name[ 8: 11]
            # parse for the location of the tif
            if int(patient) < 20:
                name='center_0/patient_'+patient+'/'+name
            elif int(patient) < 40:
                name='center_1/patient_'+patient+'/'+name
            elif int(patient) < 60:
                name='center_2/patient_'+patient+'/'+name
            elif int(patient) < 80:
                name='center_3/patient_'+patient+'/'+name
            else:
                name='center_4/patient_'+patient+'/'+name

            # preporocessing
            toadd = searchAndUpsample( i , f_name=name,  warpDim= warpDim , numRegions=numRegions , disp=False , debug=False )

            if not quiet:
                print ( name )

            arr = toadd
            
            lables = np.array( r[ind][1:])

            comb = np.array( arr , dtype = np.float32)
            return comb,np.array(lables , dtype=np.float32)
getInput.ind = 0 


# same as above just for testing.
# can probably fix this into one function
def getInputTest( f_name , i , numRegions , warpDim , quiet=True):

    with open( f_name , 'r' ) as infile:
        r = csv.reader( infile , delimiter=',' )
        r = list( r )

        lock = threading.Lock()
        lock.acquire()
        
        ind = getInputTest.ind
        getInput.ind += 1
        lock.release()

        
        name = r[ind][0]
        patient = name[ 8: 11]

        if int(patient) < 20:
            name='center_0/patient_'+patient+'/'+name
        elif int(patient) < 40:
            name='center_1/patient_'+patient+'/'+name
        elif int(patient) < 60:
            name='center_2/patient_'+patient+'/'+name
        elif int(patient) < 80:
            name='center_3/patient_'+patient+'/'+name
        else:
            name='center_4/patient_'+patient+'/'+name
            
        toadd = searchAndUpsample( i , f_name=name,  warpDim= warpDim , numRegions=numRegions , disp=False , debug=False )

        if not quiet:
            print ( name )

        arr = toadd
            
        lables = np.array( r[ind][1:])


        comb = np.array( arr , dtype = np.float32)
        return comb,np.array(lables , dtype=np.float32)


getInputTest.ind = 0


if __name__ == '__main__':


    patches = searchAndUpsample( 1 , 'center_2/patient_048/patient_048_node_4.tif' , None , 5 , delta_level=3 , disp=True )
    '''
    print ( 'done intial search' )
    for j in range( 5 ):
        patch = patches[j,:,:]

        s = SelectiveSearch()
        regions = s.search( patch )
        regions = regions[:5]
        for i in regions:
            x = i['rect'][0]
            y = i['rect'][1]
            dx= int( i['rect'][2] + x )
            dy= int( i['rect'][3] + y )

        plt.figure()
        plt.imshow( patch[ x:dx , y:dy ] )

        print ( 'done ' , i )
    '''
        
    plt.show()
