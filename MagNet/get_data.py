'''
This script will be used to load mini-batches of data.
'''


from PIL import Image
import numpy as np
import MagNet as mn

# =================================================================================================

class DataProcessor(object):
    '''
    A class to handle processing of data.
    '''
    
    def __init__(self,datadir,m=25,numpix_side=192,pixel_size=0.04,bad_files_list=None,bad_test_files_list=None,downsample=1,max_noise_rms=0.0,use_psf=False,lens_model_error=[0.01,0.01,0.01,0.01,0.01,0.01,0.01],binpix=1,mask=False,min_unmasked_flux=1.0,Interferometer=False,antennaconfig=None):
        '''
        Initialize an instance of the class.  Give it the directory
        of the directories containing training/test data.
        '''
        self.datadir = datadir
        self.num_datadir = len(datadir)
        self.pixel_size = pixel_size
        self.numpix_side = numpix_side
        self.m = m
        self.Interferometer = Interferometer
        self.antennaconfig = antennaconfig

        # create x and y to be loaded
        self.X = np.zeros([m,numpix_side**2])
        self.Y = np.zeros([m,numpix_side**2/downsample**2])

        self.Xtest = np.zeros([10*m,numpix_side**2])
        self.Ytest = np.zeros([10*m,numpix_side**2/downsample**2])
        
        self.tgi = np.zeros(10*m,dtype=bool)
        self.good_image = np.ones(m,dtype=bool)
        self.lens_model = np.zeros([m,7])
        self.test_lens_model = np.zeros([10*m,7])
        self.lens_model_error = np.zeros([m,7])
        self.test_lens_model_error = np.zeros([10*m,7])

        self.lens_models_train = np.loadtxt(datadir[0]+'parameters_train.txt')
        self.lens_models_test  = np.loadtxt(datadir[0]+'parameters_test.txt' )
        
        if bad_files_list is not None:
            self.bad_files = np.load(bad_files_list)
        else:
            self.bad_files = np.array([-1])
        
        if bad_test_files_list is not None:
            self.bad_test_files = np.load(bad_test_files_list)
        else:
            self.bad_test_files = np.array([-1])

        self.ds = downsample
        self.max_noise_rms = max_noise_rms

        self.lens_model_error_rms = np.array(lens_model_error)

        if Interferometer is False:

            if use_psf == True:
                self.binpix = binpix
                self.use_psf = use_psf
                self.psf = np.zeros([m,self.binpix*numpix_side/4+1,self.binpix*numpix_side/4+1,1,1])
                self.psf_test = np.zeros([10*m,self.binpix*numpix_side/4+1,self.binpix*numpix_side/4+1,1,1])

            self.use_mask = mask
            if self.use_mask == True:
                self.mask = np.ones([m,numpix_side,numpix_side,1])
                self.mask_test = np.ones([10*m,numpix_side,numpix_side,1])

                # Coordinates to use to get pixel locations
                self.xv,self.yv = np.meshgrid(np.arange(-self.numpix_side/2*self.pixel_size,self.numpix_side/2*self.pixel_size,self.pixel_size),np.arange(-self.numpix_side/2*self.pixel_size,self.numpix_side/2*self.pixel_size,self.pixel_size))

                # How much flux to never mask 
                self.min_unmasked_flux = 0.75

        else:
            self.UVGRID = np.zeros([self.m,2*self.numpix_side,2*self.numpix_side,1])
            self.noise  = np.zeros([self.m,2*self.numpix_side,2*self.numpix_side,1])
            self.UVGRID_test = np.zeros([10*self.m,2*self.numpix_side,2*self.numpix_side,1])
            self.noise_test = np.zeros([10*self.m,2*self.numpix_side,2*self.numpix_side,1])
    def add_uncorrelated_gaussian_noise(self,image):
        noise_rms = np.random.uniform(self.max_noise_rms/100.,self.max_noise_rms)
        noise = np.random.normal(0.0,noise_rms*np.max(image),image.shape)
        im = image + noise
        return im

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
    
    def gen_masks(self,nmax,ARCS , apply_prob=0.5):
        L_side = self.numpix_side * self.pixel_size
        mask = 1.0
        if np.min(ARCS)<0.1 and np.max(ARCS)>0.9:
            if np.random.uniform(low=0, high=1)<=apply_prob:
                while True:
                    mask = np.ones((self.numpix_side,self.numpix_side),dtype='float32')
                    num_mask = np.random.randint(1, high = nmax)
                    for j in range(num_mask):
                        x_mask =  np.random.uniform(low=-L_side/2.0, high=L_side/2.0)
                        y_mask =  np.random.uniform(low=-L_side/2.0, high=L_side/2.0)
                        r_mask = np.sqrt( (self.xv- x_mask  )**2 + (self.yv- y_mask )**2 )
                        mask_rad = 0.2
                        mask = mask * np.float32(r_mask>mask_rad)
                    if np.sum(mask*ARCS) >= (self.min_unmasked_flux * np.sum(ARCS)):
                        break
        return mask
    
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

    def generate_random_psf(self,max_width,min_width=0.0001):
        '''
        This function generates a numpix/4+1 by numpix/4+1 psf realization which will be used to smear the
        observed image during training (and it is used in the forward model).
        '''
        
        width = np.random.uniform(min_width,max_width)
        N     = self.numpix_side / 8 
        dx    = self.pixel_size / float(self.binpix)
        bp    = self.binpix
        # coordinates to set up the gaussian
        x     = np.arange(-N * dx * bp ,N * dx * bp + dx , dx)
        y     = np.arange(-N * dx * bp ,N * dx * bp + dx , dx)
        x , y = np.meshgrid(x , y)
        
        psf   = np.exp(-0.5*(x**2+y**2)/width**2)
        psf  /= np.sum(psf)

        return psf.reshape([2*N*bp+1,2*N*bp+1,1,1])

    def load_data_batch(self,max_file_num,train_or_test):
        '''
        Load a mini-batch of data into arrays X and Y.
        Data conventiion is such that it can be immediately
        fed to tensorflow (i.e. shape = (m,nx)).  The additional
        arguments indicate how many training files to draw from,
        and whether to load training data, or dev/test data.
        '''
        
        if train_or_test =='test':
            # load test data.  This means we don't want it to be random anymore
            inds = np.setdiff1d(np.arange(max_file_num),self.bad_test_files)[range(10*self.m)] 
            for i in range(10*self.m):
                file_path = np.random.choice(self.datadir)
                file_path_X = file_path+train_or_test+'_'+"%07d"%(inds[i]+1)+'.png'
                file_path_Y = file_path+train_or_test+'_'+"%07d"%(inds[i]+1)+'_source.png'
            
                # Load images.  If src is not the proper size, lets interpolate it so that it is :)
                img = np.array(Image.open(file_path_X),dtype='float32')/65535.0
                src = np.array(Image.open(file_path_Y).resize((self.numpix_side,self.numpix_side,),resample=Image.BILINEAR),dtype='float32')/65535.0 
            
                # Downsample the source if requested
                src = src.reshape(src.shape[0]/self.ds,self.ds,src.shape[0]/self.ds,self.ds).sum(axis=(1,3))
                
                if self.Interferometer is False:

                    if self.use_mask:
                        mask = self.gen_masks(10,img)
                        if isinstance(mask,np.ndarray):
                            self.mask_test[i,:] = mask.reshape(self.numpix_side,self.numpix_side,1)
                        else:
                            self.mask_test[i,:] = np.ones([self.numpix_side,self.numpix_side,1])
                    self.psf_test[i,:,:,:,:] = self.generate_random_psf(0.0001,0.1)
                else:
                    # If we're using interferometer data, we'll have to get uvgrids, and noise 
                    UVGRID,u,v = mn.get_new_UVGRID_and_db(self.pixel_size,self.numpix_side*2,deposit_order=0,antennaconfig=self.antennaconfig)
                    noise_scl = np.random.uniform(self.max_noise_rms/10.,self.max_noise_rms)
                    dim_telescope = np.fft.ifft2(np.fft.fft2(np.pad(img,[[self.numpix_side/2,self.numpix_side/2],\
                                                                         [self.numpix_side/2,self.numpix_side/2]],mode='constant',
                                                                    constant_values=0.))*np.fft.fftshift(UVGRID>0)/192.).real
                    noise_realization = np.random.normal(0.0,1.0,UVGRID.shape)
                    noise_dty = np.fft.ifft2(np.fft.fft2(noise_realization)/np.sqrt(np.fft.fftshift(UVGRID)+10**-8)*\
                                                 (np.fft.fftshift(UVGRID>0))/384.).real

                    noise = noise_realization.reshape(1,2*self.numpix_side,2*self.numpix_side,1) * np.max(dim_telescope)/np.std(noise_dty)

                    self.UVGRID_test[i,:] = np.fft.fftshift(UVGRID).reshape((1,self.numpix_side*2,self.numpix_side*2,1))
                    self.noise_test[i,:] = noise * noise_scl

                # if train, set training data, otherwise set test data
                # randomly shift normalization, but keep physics intact
                normshift = 1.-abs(1-np.random.normal(1.0,0.01))
                self.Xtest[i,:] = self.add_uncorrelated_gaussian_noise(img / np.max(img) * normshift).ravel()
                self.Ytest[i,:] = src.ravel() / np.max(src) * normshift
                self.test_lens_model[i,:] = self.lens_models_test[inds[i],:7]
                self.test_lens_model_error[i,:] = np.random.normal(0.0,self.lens_model_error_rms)

                # again, fix for NaN problem
                if (np.any(np.isnan(self.Xtest)) or np.any(np.isnan(self.Ytest))):
                    self.Xtest[i,:] = 0.
                    self.Ytest[i,:] = 0.
                    self.tgi[i] = False
                else:
                    self.tgi[i] = True
                
        else:
            # load training data (randomly)
            inds = np.random.choice(np.setdiff1d(np.arange(max_file_num),self.bad_files-1),size = self.m,replace=False)
            for i in range(self.m):
            
                file_path = np.random.choice(self.datadir)
                file_path_X = file_path+train_or_test+'_'+"%07d"%(inds[i]+1)+'.png'
                file_path_Y = file_path+train_or_test+'_'+"%07d"%(inds[i]+1)+'_source.png'
            
                # Load images.  If src is not the proper size, lets interpolate it so that it is :)
                img = np.array(Image.open(file_path_X),dtype='float32')/65535.0
                src = np.array(Image.open(file_path_Y).resize((self.numpix_side,self.numpix_side,),resample=Image.BILINEAR),dtype='float32')/65535.0 
            
                # Downsample the source if requested                                                                       
                src = src.reshape(src.shape[0]/self.ds,self.ds,src.shape[0]/self.ds,self.ds).sum(axis=(1,3))
                
                if self.Interferometer is False:
                    # Generate mask (with 50% probability) if requested
                    if self.use_mask:
                        mask = self.gen_masks(10,img)
                        if isinstance(mask,np.ndarray):
                            self.mask[i,:] = mask.reshape(self.numpix_side,self.numpix_side,1)
                        else:
                            self.mask[i,:] = np.ones([self.numpix_side,self.numpix_side,1])
                    self.psf[i,:,:,:,:] = self.generate_random_psf(0.0001,0.1)
                else:
                    # If we're using interferometer data, we'll have to get uvgrids, and noise             
                    UVGRID,u,v = mn.get_new_UVGRID_and_db(self.pixel_size,self.numpix_side*2,deposit_order=0,antennaconfig=self.antennaconfig)
                    noise_scl = np.random.uniform(self.max_noise_rms/10.,self.max_noise_rms)
                    dim_telescope = np.fft.ifft2(np.fft.fft2(np.pad(img,[[self.numpix_side/2,self.numpix_side/2],\
                                                                         [self.numpix_side/2,self.numpix_side/2]],mode='constant',
                                                                    constant_values=0.))*np.fft.fftshift(UVGRID>0)/192.).real
                    noise_realization =np.random.normal(0.0,1.0,UVGRID.shape)
                    noise_dty =np.fft.ifft2(np.fft.fft2(noise_realization)/np.sqrt(np.fft.fftshift(UVGRID)+10**-8)*\
                                                 (np.fft.fftshift(UVGRID>0))/384.).real

                    noise = noise_realization.reshape(1,2*self.numpix_side,2*self.numpix_side,1) * np.max(dim_telescope)/np.std(noise_dty)
                    self.UVGRID[i,:] = np.fft.fftshift(UVGRID).reshape((1,self.numpix_side*2,self.numpix_side*2,1))
                    self.noise[i,:] = noise * noise_scl

                # if train, set training data, otherwise set test data
                # randomly shift normalization, but keep physics intact
                normshift = 1.-abs(1-np.random.normal(1.0,0.01))
                self.X[i,:] = self.add_uncorrelated_gaussian_noise(img / np.max(img) * normshift).ravel()
                self.Y[i,:] = src.ravel() / np.max(src) * normshift
                self.lens_model[i,:] = self.lens_models_train[inds[i],:7]
                self.lens_model_error[i,:] = np.random.normal(0.0,self.lens_model_error_rms)

                # temporary fix for NaN problem --> set them to 0.
                if (np.any(np.isnan(self.X)) or np.any(np.isnan(self.Y))):
                    self.X[i,:] = 0.
                    self.Y[i,:] = 0.
                    self.good_image[i] = False
                else:
                    self.good_image[i] = True

#        print np.any(np.isnan(self.X)),np.any(np.isnan(self.Y))
            
            
        return
        
    def get_gridded_visibilities(self,vis_file_list,oversample_uv=1,phase_center_shifts=None,newRipples=False):
    
        # Lets make some shorthands
        ns = self.numpix_side
        ouv = oversample_uv
        Nvis = len(vis_file_list)

        # Array for input visibilities
        self.Vis_input = np.zeros([Nvis,ns*ouv,ns*ouv,1])
        self.UVGRID    = np.zeros([Nvis,ns*ouv,ns*ouv,1])
        self.sigma     = np.zeros([Nvis,ns*ouv,ns*ouv,1])

        # If not specified, phase center shift is 0
        if phase_center_shifts is None:
            phase_center_shifts = [[0.,0.] for i in range(Nvis)]

        # Grid the visibilities
        for i in range(Nvis):
            self.UVGRID[i,:,:,0],self.Vis_input[i,:,:,0],self.sigma[i,:,:,0] = get_gridded_visibilities(vis_file_list[i],\
                                                                                                        self.pixel_size,\
                                                                                                        ns*ouv,\
                                                                                                        phase_center_shifts,\
                                                                                                        newRipples)
        return



def load_binary(binaryfile):
    with open(binaryfile,'rb') as file:
        filecontent = file.read()
        data = np.array(struct.unpack("d"*(len(filecontent)//8),filecontent))
    file.close()
    return data


def get_binned_visibilities(u,v,vis,sigma,pix_res,num_pixels):
    '''                                                                                                                                        
    convert from 1d vector of u, v, vis to a 2d histogram of uv, vis                                                                           
                                                                                                                                               
    Takes:                                                                                                                                     
                                                                                                                                               
    u:     The u coordinates of the data (in meters)                                                                                           
                                                                                                                                               
    v:     The v coordinates of the data (in meters)                                                                                           
                                                                                                                                               
    vis:   The visibility data (in Jy), complex format                                                                                         
                                                                                                                                               
    Returns:                                                                                                                                   
                                                                                                                                               
    A:     The noise scaling.                                                                                                                  
                                                                                                                                               
    '''

    kvec = np.fft.fftshift(np.fft.fftfreq(num_pixels,pix_res/3600./180.*np.pi))
    kvec -= (kvec[1]-kvec[0])/2.
    kvec = np.append(kvec,kvec[-1]+(kvec[1]-kvec[0]))

    print np.sum(np.isclose((kvec[1:]+kvec[:-1])/2.,0))

    # Count number of visibilities in each bin                                                                                                 
    P,reject1,reject2 = np.histogram2d(u,v,bins=kvec)
    P2,reject1,reject2 = np.histogram2d(-u,-v,bins=kvec)
    vis_gridded = np.zeros(P.shape,dtype=complex)
    sigma_gridded = np.zeros(P.shape,dtype=float)

    # Keep only bins that contain visibilities                                                                                                 
    [row,col] = np.where(P!=0)
    [row2,col2] = np.where(P2!=0)

    # Keep track of stats (just in case something weird is happening)
    NumSkippedBins = 0
    TotalUsed      = 0

    # Array for the indices of the visibilities that are subtracted
    indI = np.zeros(u.shape,int)

    # loop over bins
    for i in range(len(row)):

        # indices of visibilities in the bin                                                                                                   
        inds = np.where((v>=kvec[col[i]]) & (v<kvec[col[i]+1]) & \
                        (u>=kvec[row[i]]) & (u<kvec[row[i]+1]))[0]



        vis_gridded[col[i],row[i]] =np.average(vis[inds],weights = sigma[inds]**-2.)
        sigma_gridded[col[i],row[i]] = np.sum(sigma[inds]**-2.)**-0.5

    for i in range(len(row2)):

        # indices of visibilities in the bin                                                                                                   
        inds = np.where((-v>=kvec[col2[i]]) & (-v<kvec[col2[i]+1]) & \
                        (-u>=kvec[row2[i]]) & (-u<kvec[row2[i]+1]))[0]


        # This time, lets store the vis and sigma
        vis_avg = np.average(np.conj(vis[inds]),weights=sigma[inds]**-2.)
        sigma_avg = np.sum(sigma[inds]**-2.)**-0.5

        if not np.isclose(vis_gridded[col2[i],row2[i]],0,rtol=1e-8.):
            vg = vis_gridded[col2[i],row2[i]]
            sg = sigma_gridded[col2[i],row2[i]]

            vis_gridded[col2[i],row2[i]] = np.average([vis_avg,vg],weights=[sigma_avg**-2,sg**-2.])
            sigma_gridded[col2[i],row2[i]] = 1./np.sqrt(sg**-2+sigma_avg**-2)
        else:
            vis_gridded[col2[i],row2[i]] = vis_avg
            sigma_gridded[col2[i],row2[i]] = sigma_avg

    # get average by division                                                                                                                  
    #vis_gridded[np.where(P+P2 !=0)] /= (P+P2)[np.where(P+P2 !=0)].astype('float')                                                             

    vis_gridded /= (P.T+P2.T+1e-8)
    vis_gridded[np.abs(vis_gridded)<1e-6] *=0
    vis_gridded[np.abs(vis_gridded)>1e7]  *=0
    
    return (P+P2).T , vis_gridded


def get_gridded_visibilities(directory_name,pix_res,num_pixels,phasecenter=[0.,0.],newRipples=False):
    '''                                                                                                                                      
    Load visibilities from a file, and then produce the gridded (averaged in grid cells)                                                     
    visibilities and uv mask that can be fed to the likelihood object)                                                                       
    '''
    # first lets load the data                                                                                                               
    u = load_binary(directory_name+'u.bin')
    v = load_binary(directory_name+'v.bin')
    vis = load_binary(directory_name+'vis_chan_0.bin')
    vis = vis[::2]+1j*vis[1::2]
    sig = load_binary(directory_name+'sigma_squared_inv.bin')[::2]**-0.5

    if newRipples is True:
        freq = load_binary(directory_name+'frequencies.bin')
        wav =  (3.*10**8) / freq
        u /= wav
        v /= wav

    # If desired, shift the phase center so that the image lies in the Field of view.
    vis = shift_phase_center(u,v,vis,phasecenter)

    # Grid the visibilities
    UVGRID , vis_gridded , sigma = get_binned_visibilities(u,v,vis,sig,pix_res,num_pixels)

    UVGRID = np.fft.fftshift(UVGRID).reshape([1,num_pixels,num_pixels,1])
    vis_gridded = np.fft.fftshift(vis_gridded).reshape([1,num_pixels,num_pixels,1])
    sigma = np.fft.fftshift(sigma).reshape([1,num_pixels,num_pixels,1])

    return UVGRID, vis_gridded, sigma


def shift_phase_center(u,v,vis,phase_center):
    '''                                                                                                                            
    Shift the center of the ALMA pointing to a new phase center.  Phase center shift 
    is defined in arcseconds.                                        
    '''
    ushift = phase_center[0]*u / 3600. / 180. * np.pi
    vshift = phase_center[1]*v / 3600. / 180. * np.pi
    phaseshift = ushift + vshift

    vis_shifted = vis * np.exp(2j*np.pi*phaseshift)
    return vis_shifted
