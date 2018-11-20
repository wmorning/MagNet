import tensorflow as tf
import numpy as np



class SrcLikelihood(object):
    '''
    This class will perform the log likelihood on a batch of images given a list of their parameters.
    '''
    
    def __init__(self,Y,img=None,numpix_side=192,pix_res=0.04,numpix_src=192/4,src_res=0.01,psf=False,psf_pl=None,mask_pl=None,poisson=False):
        '''
        Initialize the class and set the parameters
        '''
        self.lens_model = Y
#        self.lens_model_predicted = Ypred


        if img is not None:
            self.img = img
        else:
            self.image_included = False

        self.numpix_side = numpix_side
        self.pix_res     = pix_res
        self.numpix_src  = numpix_src
        self.src_res     = src_res
        self.psf = psf
        if self.psf: self.psf_pl = psf_pl
        if mask_pl is not None: self.mask_pl = mask_pl
        self.poisson = poisson

        self.te = tf.transpose(tf.slice(self.lens_model,[0,0],[-1,1]))
        self.ex = tf.transpose(tf.slice(self.lens_model,[0,1],[-1,1]))
        self.ey = tf.transpose(tf.slice(self.lens_model,[0,2],[-1,1]))
        self.xl = tf.transpose(tf.slice(self.lens_model,[0,3],[-1,1]))
        self.yl = tf.transpose(tf.slice(self.lens_model,[0,4],[-1,1]))
        self.gx = tf.transpose(tf.slice(self.lens_model,[0,5],[-1,1]))
        self.gy = tf.transpose(tf.slice(self.lens_model,[0,6],[-1,1]))
        
        self.grid = tf.transpose(self._meshgrid(numpix_side,pix_res))
        
    def Add_predicted_lens_model(self,lensmodel_pred):
        self.lens_model_predicted = lensmodel_pred

        self.tep = tf.transpose(tf.slice(self.lens_model_predicted,[0,0],[-1,1]))                                                       
        self.exp = tf.transpose(tf.slice(self.lens_model_predicted,[0,1],[-1,1]))                                                      
        self.eyp = tf.transpose(tf.slice(self.lens_model_predicted,[0,2],[-1,1]))                                                      
        self.xlp = tf.transpose(tf.slice(self.lens_model_predicted,[0,3],[-1,1]))                                                     
        self.ylp = tf.transpose(tf.slice(self.lens_model_predicted,[0,4],[-1,1]))                                                
        self.gxp = tf.transpose(tf.slice(self.lens_model_predicted,[0,5],[-1,1]))                                               
        self.gyp = tf.transpose(tf.slice(self.lens_model_predicted,[0,6],[-1,1]))
        
    def Loglikelihood(self,src,img=None):
        '''
        Computes the likelihood of observing an image (img), given a model image for the src
        (src).  Computes the raytracing over multiple images in a vectorized way.  
        '''
        
        self.src = src
        
        xsrc , ysrc = self.raytrace2()
        
        xsrc = tf.reshape(xsrc,[-1])
        ysrc = tf.reshape(ysrc,[-1])
        
        img_pred = self._interpolate(self.src,xsrc,ysrc,[self.numpix_side,self.numpix_side],self.src_res)
        img_pred = tf.reshape(img_pred,tf.shape(self.img))

        if self.psf:
            img_pred = tf.nn.conv2d(img_pred,self.psf_pl,strides=[1,1,1,1],padding='SAME')

        if hasattr(self,'mask_pl'):
            img_pred = tf.multiply(img_pred,self.mask_pl)
        mse = tf.reduce_sum(tf.square(tf.subtract(img_pred,self.img)))
        
        if hasattr(self,'noise_rms'):
            mse /= tf.square(self.noise_rms)
        
        return mse 
    
    def build_image(self,src_true,noisy=False,max_noise_rms=0.1):
        '''
        This function is intended to construct the graph that builds the 
        observed image without being given it as a placeholder.  It should
        be called after __init__ but before any other methods.
        
        It accepts a placeholder for the source image as an input.
        '''
        self.img = self.Predict_image(src_true,noisy,max_noise_rms)
    
    def Predict_image(self,src,noisy=False,max_noise_rms=0.1):
        '''
        Given an input for the source image (and a lens model),
        return an output image corresponding to the lens model raytraced
        image.
        '''
        self.src = src
        xsrc , ysrc = self.raytrace()
        
        xsrc = tf.reshape(xsrc,[-1])
        ysrc = tf.reshape(ysrc,[-1])

        img_pred = self._interpolate(self.src,xsrc,ysrc,[self.numpix_side,self.numpix_side],self.src_res)
        img_pred = tf.reshape(img_pred,[-1,self.numpix_side,self.numpix_side,1])
        
        if self.psf == True:
            #img_pred = batch_conv(img_pred,self.psf_pl)
            img_pred = tf.nn.conv2d(img_pred,self.psf_pl,strides=[1,1,1,1],padding='SAME')
        
        if self.poisson:
            counts = tf.random_uniform([1,1,1,1],minval=100,maxval=1000)
            img_pred = tf.random_poisson(counts*img_pred,[1])[0]/counts

        if noisy ==True:
            noise_rms = tf.random_uniform(shape=[1],minval=max_noise_rms/100.,maxval=max_noise_rms)
            noise = tf.random_normal(tf.shape(img_pred),mean=0.0,stddev = noise_rms)
            img_pred = tf.add(img_pred,noise)
            self.noise_rms = noise_rms
        return img_pred

    def Predict_predicted_image(self,src):
        '''                                                                                                      
        Given an input for the source image (and a lens model),                                                  
        return an output image corresponding to the lens model raytraced                                         
        image.                                                                                                   

        Noisy is not an option here, since we're predicting the predicted image, and thus
        it should be noiseless.
        '''

        xsrc , ysrc = self.raytrace2()

        xsrc = tf.reshape(xsrc,[-1])
        ysrc = tf.reshape(ysrc,[-1])

        img_pred = self._interpolate(src,xsrc,ysrc,[self.numpix_side,self.numpix_side],self.src_res)
        img_pred = tf.reshape(img_pred,[-1,self.numpix_side,self.numpix_side,1])

        if self.psf == True:
            img_pred = tf.nn.conv2d(img_pred,self.psf_pl,strides=[1,1,1,1],padding='SAME')

        return img_pred


    def _repeat(self,x, n_repeats):
        '''
        Not sure what this does, but _interpolate calls it
        '''
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])
        
    def _meshgrid(self,numpix,pix_res):
        '''
        Create a meshed grid with numpix pixels, and a full size given
        by pix_res * numpix
        '''
        #with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([numpix, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-numpix/2*pix_res, numpix/2*pix_res, numpix), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-numpix/2*pix_res, numpix/2*pix_res, numpix), 1),
                        tf.ones(shape=tf.stack([1, numpix])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat( [x_t_flat, y_t_flat, ones], 0)
        return grid
            
    def _interpolate(self,im, x, y, out_size,src_res):
        '''
        Interpolate the input image (im) at coordinates x and y.
        This also takes the size of the output image and resolution
        of the input image.  This is necessary because it defines
        the scaling between source and image.
        '''
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]
        
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        #x = (x + 1.0)*(width_f) / 2.0
        #y = (y + 1.0)*(height_f) / 2.0
        x = (x + width_f*src_res/2.) / src_res
        y = (y + height_f*src_res/2.) / src_res

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = self._repeat(tf.range(num_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return output


    def raytrace(self):
        '''
        Given the predicted lens model parameters fed to the likelihood function object, 
        calculate the raytraced pixels in the source plane.  For now, we assume 
        an SIE for the lens model.  
        '''
    
        q = 1-tf.sqrt(tf.add(tf.square(self.ex),tf.square(self.ey)))
        qp = tf.sqrt(1-tf.square(q))
        angle = atan2(self.ey,self.ex)
        shear = tf.sqrt(tf.square(self.gx)+tf.square(self.gy))
        shearang = tf.subtract(atan2(self.gy,self.gx) , angle)
        g1 = tf.multiply(shear,-tf.cos(tf.scalar_mul(2,shearang)))
        g2 = tf.multiply(shear,-tf.sin(tf.scalar_mul(2,shearang)))
        g3 = tf.multiply(shear,-tf.sin(tf.scalar_mul(2,shearang)))
        g4 = tf.multiply(shear, tf.cos(tf.scalar_mul(2,shearang)))
    
        xgrid = tf.slice(self.grid,[0,0],[-1,1])
        ygrid = tf.slice(self.grid,[0,1],[-1,1])
        
        xgrid = tf.subtract(xgrid,self.xl)
        ygrid = tf.subtract(ygrid,self.yl)
        
        rad , th = cart2pol(ygrid,xgrid)
        xgrid,ygrid = pol2cart(rad,tf.subtract(th,angle))
        
        par = atan2(ygrid,xgrid)
    
        xsrc = tf.subtract(xgrid , tf.multiply(self.te , tf.multiply(tf.divide(tf.sqrt(q) , qp) , asinh( tf.divide(tf.multiply(qp , tf.cos(par)) , q)))))
        ysrc = tf.subtract(ygrid , tf.multiply(self.te , tf.multiply(tf.divide(tf.sqrt(q) , qp) , tf.asin(tf.multiply(qp , tf.sin(par))))))
        xsrc = tf.subtract(xsrc,tf.add(tf.multiply(g1,xgrid),tf.multiply(g2,ygrid)))
        ysrc = tf.subtract(ysrc,tf.add(tf.multiply(g3,xgrid),tf.multiply(g4,ygrid)))
        
        rad , th = cart2pol(ysrc,xsrc)
        xsrc,ysrc = pol2cart(rad,tf.add(th,angle))
        xsrc = tf.add(xsrc,self.xl)
        ysrc = tf.add(ysrc,self.yl)
        
        
        return tf.transpose(xsrc) , tf.transpose(ysrc)


    def raytrace2(self):
        q = 1-tf.sqrt(tf.add(tf.square(self.exp),tf.square(self.eyp)))
        qp = tf.sqrt(1-tf.square(q))
        angle = atan2(self.eyp,self.exp)
        shear = tf.sqrt(tf.square(self.gxp)+tf.square(self.gyp))
        shearang = tf.subtract(atan2(self.gyp,self.gxp) , angle)
        g1 = tf.multiply(shear,-tf.cos(tf.scalar_mul(2,shearang)))
        g2 = tf.multiply(shear,-tf.sin(tf.scalar_mul(2,shearang)))
        g3 = tf.multiply(shear,-tf.sin(tf.scalar_mul(2,shearang)))
        g4 = tf.multiply(shear, tf.cos(tf.scalar_mul(2,shearang)))
    
        xgrid = tf.slice(self.grid,[0,0],[-1,1])
        ygrid = tf.slice(self.grid,[0,1],[-1,1])
        
        xgrid = tf.subtract(xgrid,self.xlp)
        ygrid = tf.subtract(ygrid,self.ylp)
        
        rad , th = cart2pol(ygrid,xgrid)
        xgrid,ygrid = pol2cart(rad,tf.subtract(th,angle))
        
        par = atan2(ygrid,xgrid)
    
        xsrc = tf.subtract(xgrid , tf.multiply(self.tep , tf.multiply(tf.divide(tf.sqrt(q) , qp) , asinh( tf.divide(tf.multiply(qp , tf.cos(par)) , q)))))
        ysrc = tf.subtract(ygrid , tf.multiply(self.tep , tf.multiply(tf.divide(tf.sqrt(q) , qp) , tf.asin(tf.multiply(qp , tf.sin(par))))))
        xsrc = tf.subtract(xsrc,tf.add(tf.multiply(g1,xgrid),tf.multiply(g2,ygrid)))
        ysrc = tf.subtract(ysrc,tf.add(tf.multiply(g3,xgrid),tf.multiply(g4,ygrid)))
        
        rad , th = cart2pol(ysrc,xsrc)
        xsrc,ysrc = pol2cart(rad,tf.add(th,angle))
        xsrc = tf.add(xsrc,self.xlp)
        ysrc = tf.add(ysrc,self.ylp)
        
        return tf.transpose(xsrc) , tf.transpose(ysrc)
                 
def cart2pol(y,x):
    '''
    Convert from cartesian to polar coordinates (using tensors)
    '''
    return tf.sqrt(tf.add(tf.square(y),tf.square(x))),atan2(y,x)

def pol2cart(r,th):
    '''
    convert from polar to cartesian coordinates (using tensors)
    '''
    return tf.multiply(r,tf.cos(th)) , tf.multiply(r,tf.sin(th))
                         
def atan2(y,x):
    ''' 
    because tensorflow didn't add it until later >:(
    '''
    tan1 = tf.atan(tf.divide(y,x))
    tan1 = tf.add(tan1,tf.multiply(tf.multiply(tf.sign(y),tf.constant(np.pi)),tf.cast(tf.less(x,0),tf.float32)))
    return tan1

def asinh(x):
    '''
    tensorflow is worthless
    '''
    return tf.log(tf.add(x,tf.sqrt(tf.add(tf.square(x),tf.constant(1.0)))))


def single_conv(tupl):
    '''
    Helps us to do a batch-wise convolution (with the psf placeholder)
    which in turn lets us apply different psfs within a batch
    
    Give it a tupl containing the tensor for the image and the tensor for
    the psf.
    '''
    IM , PSF = tupl
    return tf.nn.conv2d(IM , PSF , strides=(1,1,1,1),padding='SAME')

def batch_conv(IM,PSF):
    '''
    Given m images and m psfs, this function convolves each image
    with its associated psf.  This allows us to train using
    different psfs within a single batch, which is probably useful 
    to make training more efficient.
    '''
    return tf.squeeze(tf.map_fn(single_conv,(tf.expand_dims(IM,1),PSF),dtype=tf.float32),axis=1)
