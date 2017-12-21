# Copyright Jonathan Booher 2017. All rights reserverd

import multiresolutionimageinterface as mir
from PIL import Image
import PIL
import time

from matplotlib import pyplot as plt 

from skimage.segmentation import *
import numpy as np
import skimage
from skimage.feature import local_binary_pattern


# note these includes are not used right now but may be in the future
from numba import guvectorize
from numba import float64

class SelectiveSearch():
    #note constants is used to provide weighting to the similarity measures
    def __init__( self , scale=250 , sigma=0.98 , min_size=50 , min_ratio=0. , max_ratio=0.8 , thresh_dist=50 ,
                  constants=[1.0,1.0,1.0,1.0] ):
        self.scale =scale
        self.sigma = sigma
        self.min_size = min_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.thresh_dist = thresh_dist
        self.constants = constants

        return
    # similarity metrics as defined in the original selective search paper
    def calc_similiarity( self , r1 , r2 , imgsize ):
        def s_color():
            return sum( [min( a, b ) for a , b in zip( r1['hist_c'] , r2['hist_c'] ) ] )
        def s_texture():
            return sum( [min( a, b ) for a , b in zip( r1['hist_t'] , r2['hist_t'] ) ] )
        def s_size():
            return 1.0 - ( r1['size'] + r2['size'] )/ imgsize
        def s_fill():
            size_b_box =    ( max( r1['max_x'], r2['max_x']) - min( r1['min_x'] , r2['min_x'] ) ) * (   max( r1['max_y'], r2['max_y']) - min( r1['min_y'] , r2['min_y'] ) )

            return 1.0 - ( size_b_box - r1['size'] - r2['size'] )/ imgsize

        return np.dot( self.constants , np.array([s_color() , s_texture() , s_size() , s_fill() ] , dtype=np.float32) )


    # gen historgram
    def hist_gen( self , img , n_bins , max_val ):

        hist = np.array( [] )
        for channel in range( 0 , 3 ):
            c = img[ : , channel ]
            hist = np.concatenate( [ hist ] + [np.histogram( c , n_bins , ( 0.0 , max_val ))[0] ] )

        hist /= len( img )

        return hist

    # calc the texture gradient
    def text_grad( self, img ):
        ret = np.zeros_like( img )

        for channel in range( 0 , 3 ):
            ret[ : , : , channel ] = local_binary_pattern( img[:,:,channel] , 8 , 1.0 )

        return ret

    def extract_regions(self , img):
        R = {}

        # get hsv image
        hsv = skimage.color.rgb2hsv(img[:, :, :3])

        # iter over all the pixels so we can identify regions and calc min/max x/y for each region.
        for y, i in enumerate(img):
            for x, (r, g, b, l) in enumerate(i):

                # new region note that .
                if l not in R:
                    R[l] = { 'min_x': 0xffffffff, 'min_y': 0xffffffff, 'max_x': 0, 'max_y': 0, 'labels': [l]}

                # update the bounding box for this region if needed.
                if R[l]['min_x'] > x:  R[l]['min_x'] = x
                if R[l]['min_y'] > y:  R[l]['min_y'] = y
                if R[l]['max_x'] < x:  R[l]['max_x'] = x
                if R[l]['max_y'] < y:  R[l]['max_y'] = y

        # texture gradient need for the texture histogram.
        tex_grad = self.text_grad(img)

        for k, v in list(R.items()):

            masked_pixels = hsv[:, :, :][img[:, :, 3] == k] # only copy the values from hsv into masked_pixels if the correct region ( k )
            R[k]['size'] = len(masked_pixels / 4)

            R[k]['hist_c'] = self.hist_gen( masked_pixels , 25 , 255.0 ) 
            R[k]['hist_t'] = self.hist_gen( tex_grad[: , : ][img[:,:,3]==k] , 10 , 1.0)


        return R

    def extract_neighbors( self , regions ):
        # checks whether a and b are neighbors
        def intersect(a, b):
            if (a['min_x'] < b['min_x'] < a['max_x']
                    and a['min_y'] < b['min_y'] < a['max_y']) or (
                a['min_x'] < b['max_x'] < a['max_x']
                    and a['min_y'] < b['max_y'] < a['max_y']) or (
                a['min_x'] < b['min_x'] < a['max_x']
                    and a['min_y'] < b['max_y'] < a['max_y']) or (
                a['min_x'] < b['max_x'] < a['max_x']
                    and a['min_y'] < b['min_y'] < a['max_y']):
                return True
            return False

        # .items returns tuple of the index and the descritption of the region at that index
        R = list(regions.items())

        neighbors = []
        for cur, a in enumerate(R[:-1]):
            for b in R[cur + 1:]:
                if intersect(a[1], b[1]): # get the region description
                    neighbors.append((a, b)) # append tuple of tuples

        return neighbors

    def merge_regions( self , r1 , r2 ):
        new_size = r1['size']+r2['size']

        return  { 'min_x': min(r1['min_x'], r2['min_x']), 'min_y': min(r1['min_y'], r2['min_y']),
                  'max_x': max(r1['max_x'], r2['max_x']), 'max_y': max(r1['max_y'], r2['max_y']),
                  'size': new_size, 'labels': r1['labels'] + r2['labels'],
                  'hist_c': ( r1['hist_c'] * r1['size'] + r2['hist_c'] * r2['size']) / new_size,
                  'hist_t': ( r1['hist_t'] * r1['size'] + r2['hist_t'] * r2['size']) / new_size
                  }

    def search( self , img_orig ):
        assert img_orig.shape[2] == 3 , 'must be 3 channel'


        # try to improve efficiency of initial rough segmentation and extract regions.
        # those are the two functions that take all the time in this implementation of selective_search

        # initial rough segmentation
        img_mask = skimage.segmentation.felzenszwalb(
            skimage.util.img_as_float(img_orig), scale=self.scale, sigma=self.sigma,
            min_size=self.min_size)

        img = np.append( img_orig, np.zeros(img_orig.shape[:2])[:, :, np.newaxis], axis=2)
        img[:, :, 3] = img_mask
        # ***************************


        if img is None: # no intital segments found.
            return None, {}

        img_size = img.shape[0] * img.shape[1]

        R = self.extract_regions( img )
        neighbors = self.extract_neighbors( R )

        S = {}
        for ( ind_a , region_a ), ( ind_b , region_b ) in neighbors:
            S[ (ind_a,ind_b) ] = self.calc_similiarity( region_a , region_b , img_size )

        while S != {}:

            r_i, r_j= sorted( S.items(), key=lambda i: i[1])[-1][0]

            new_index = max(R.keys() ) + 1.0
            R[new_index] = self.merge_regions( R[r_i] , R[r_j] )

            to_del = []
            for k,v in list( S.items() ):
                if r_i in k or r_j in k: to_del.append( k ) # append the key if either of the merged regions are in the pair
            for k in to_del: del S[k] # remove all the ones that we found. Note this has to be done seperately bc changing len(S)
            

            # the neighbors of the new region will be the interesection of region i and j minus ( i, j )
            # so we only need to calculate the similarity with those.
            for k in [ a for a in to_del if a != ( r_i,r_j ) ]:
                pair = k[1] if k[0] in ( r_i,r_j ) else k[0] # make sure we use a valid key for indexing into S
                
                S[ ( new_index  , pair ) ] = self.calc_similiarity( R[new_index] , R[pair] , img_size )

        regions = []
        min_pts = []

        # note uncomment all mentions of patches below to return the sections of the images identified
        #patches = []

        for k , r in list(R.items() ):

            # use this to eliminate nearly identical regions.
            flg = True
            for pt in min_pts:
                if ( pt[0] - r['min_x'] )**2 + ( pt[1] - r['min_y'] ) **2 < self.thresh_dist**2:
                    flg = False
                    break

            r['size'] = (r['max_x'] - r['min_x']) * (r['max_y'] - r['min_y'])
            if r['size'] > self.min_ratio*img_size and r['size'] < self.max_ratio*img_size and flg == True:
                min_pts.append( ( r['min_x'] , r['min_y'] ) )
                regions.append({
                    'rect': (  r['min_x'], r['min_y'],  r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
                    'size': r['size'], 'labels': r['labels']
                })
                #patches.append( img_orig[   r['min_y']:r['max_y'] , r['min_x']:r['max_x'] , :    ] )

        regions= sorted( regions , key = lambda t : t['size']  , reverse=True)

        return regions #, patches

# sample usage
if __name__ == '__main__':

    start = time.time()

    file_name='center_1/patient_030/patient_030_node_1.tif'

    # mir lets us read multi resolution TIF images.
    #    the structure of the tif files is pyramidal
    reader= mir.MultiResolutionImageReader()
    level = 8 # get a small enough version of the image that we can actually efficiently calc bounding boxes on.
    mr_image = reader.open( file_name )
    dim_x, dim_y = mr_image.getLevelDimensions( level )
    ds = mr_image.getLevelDownsample(level) # get the scale factor for indexing into level 0.

    image_patch = mr_image.getUCharPatch(int(0 * ds), int(0 * ds), dim_x, dim_y , level) # get the whole image at this level.

    img = Image.fromarray(image_patch)
    img = np.array( img )

    selective_search = SelectiveSearch()
    #regions, ptches = selective_search.search( img )
    regions = selective_search.search( img )

    
    print ( 'Preprocessing pipeline takes:\t ' + str( time.time() - start ) )
    print ( str(len( regions )) + '   bounding boxes were found' )


    import matplotlib.patches as patches
    # show all the bounding boxes. 
    fig,ax = plt.subplots(1)
    ax.imshow( img )

    #plt.figure( 0 )
    for i in range( 0 , len(regions) ):
        ind = i
        rect = patches.Rectangle( (regions[ind]['rect'][0] , regions[ind]['rect'][1]) , regions[ind]['rect'][2] , regions[ind]['rect'][3]  , linewidth=2,edgecolor='r' , facecolor='none')

        ax.add_patch( rect )



    import copy

    level -= 2
    print ( ds )
    old_ds = copy.deepcopy(ds) 
    ds =  mr_image.getLevelDownsample( level )

    plt.figure()

    x = int( regions[0]['rect'][0] * old_ds )
    y = int( regions[0]['rect'][1] * old_ds )
    dx= int( regions[0]['rect'][2] * ds / 17 ) # 17 seems magic but is prob a constant inside MIR
    dy= int( regions[0]['rect'][3] * ds / 17 )


    img_patch = mr_image.getUCharPatch( x , y , dx , dy , level )

    plt.imshow( np.array( Image.fromarray( img_patch ).resize( ( 512 , 512 ) ) ) )
    
    
    #for i in range( 0 , len( regions ) ):
    #    plt.figure()
        
    #    plt.imshow( ptches[i] )

        
    plt.show()
