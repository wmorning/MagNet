'''
This script will be used to load mini-batches of data.
'''


from PIL import Image
import numpy as np

# =================================================================================================

class DataProcessor(object):
    '''
    A class to handle processing of data.
    '''
    
    def __init__(self,datadir,m=25,numpix_side=192,pixel_size=0.04):
        '''
        Initialize an instance of the class.  Give it the directory
        of the directories containing training/test data.
        '''
        self.datadir = datadir
        self.num_datadir = len(datadir)
        self.pixel_size = pixel_size
        self.numpix_side = numpix_side
        self.m = m
        
        # create x and y to be loaded
        self.X = np.zeros([m,numpix_side**2])
        self.Y = np.zeros([m,numpix_side**2])

    def pick_new_lens_center(self,ARCS,Y, numpix_side,pixel_size,xy_range = 0.5):
        '''
        We wont use this much yet, but we will eventually.  This function chooses a
        random position to shift the center of the lens by.  The shift is such that the 
        source flux is not moved off the pixel grid (that would be bad).  
        '''
    
        # get the current random state 
        rand_state = np.random.get_state()
    
        # This loop will require valid image shifts
        while True:
            # pick new x and y position, in pixels
            x_new = np.random.randint( -1 * np.ceil(xy_range/2/pixel_size) , high = np.ceil(xy_range/2/pixel_size) )
            y_new = np.random.randint( -1 * np.ceil(xy_range/2/pixel_size) , high = np.ceil(xy_range/2/pixel_size) )
        
            # get the new index of the central pixel
            m_shift = - int(np.floor(Y[3]/pixel_size) - x_new)
            n_shift = - int(np.floor(Y[4]/pixel_size) - y_new)
        
            # shift the image by m and n
            shifted_ARCS = im_shift(ARCS.reshape((numpix_side,numpix_side)), m_shift , n_shift ).reshape((numpix_side*numpix_side,))
        
            # if the given shift is valid, we can exit the loop
            if np.sum(shifted_ARCS) >= ( 0.98 * np.sum(ARCS) ):
                break
    
        # update the truths to reflect the shift (this may not be used in source reconstruction)   
        lensXY = np.array( [ np.double(m_shift) * pixel_size+ Y[3] , np.double(n_shift) * pixel_size + Y[4] ])
        np.random.set_state(rand_state)
        return shifted_ARCS , lensXY , m_shift, n_shift
    
    def im_shift(self,im, m , n):
        '''
        Shift the image m pixels horizontally, and n pixels vertically
        '''
        shifted_im1 = np.zeros(im.shape)
        if n > 0:
            shifted_im1[n:,:] = im[:-n,:]
        elif n < 0:
            shifted_im1[:n,:] = im[-n:,:]
        elif n ==0:
            shifted_im1[:,:] = im[:,:]
        shifted_im2 = np.zeros(im.shape)
        if m > 0:
            shifted_im2[:,m:] = shifted_im1[:,:-m]
        elif m < 0:
            shifted_im2[:,:m] = shifted_im1[:,-m:]
        shifted_im2[np.isnan(shifted_im2)] = 0
        return shifted_im2
    
    
    def load_data_batch(self,max_file_num,train_or_test):
        '''
        Load a mini-batch of data into arrays X and Y.
        Data conventiion is such that it can be immediately
        fed to tensorflow (i.e. shape = (m,nx)).  The additional
        arguments indicate how many training files to draw from,
        and whether to load training data, or dev/test data.
        
        iteration 0:  Just load the X and Y, bin the Y, and
        normalize
        '''
        
        if train_or_test =='test':
            # load test data.  This means we don't want it to be random anymore
            #inds = range(self.m)
            inds = np.random.randint(0,high=max_file_num,size = self.m)
        else:
            # load training data (randomly)
            inds = np.random.randint(0,high=max_file_num,size = self.m)
        for i in range(self.m):
            
            file_path = np.random.choice(self.datadir)
            file_path_X = file_path+train_or_test+'_'+"%07d"%(inds[i]+1)+'.png'
            file_path_Y = file_path+train_or_test+'_'+"%07d"%(inds[i]+1)+'_source.png'
            
            # Load images.  If src is not the proper size, lets interpolate it so that it is :)
            img = np.array(Image.open(file_path_X),dtype='float32')/65535.0
            src = np.array(Image.open(file_path_Y).resize((self.numpix_side,self.numpix_side,),resample=Image.BILINEAR),dtype='float32')/65535.0 
            
            
            # if train, set training data, otherwise set test data
            # randomly shift normalization, but keep physics intact
            normshift = np.random.normal(1.0,0.01)
            self.X[i,:] = img.ravel() / np.max(img) * normshift
            self.Y[i,:] = src.ravel() / np.max(src) * normshift

            # temporary fix for NaN problem --> set them to 0.
            self.X[i,np.where(np.isnan(img.ravel()))] = 0.
            self.Y[i,np.where(np.isnan(src.ravel()))] = 0.
            
            
        return
        
