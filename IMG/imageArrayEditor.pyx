
# distutils: language = c++
import cython
from numpy cimport ndarray, int_t, uint8_t
from numpy import asarray, newaxis, resize, zeros, uint8
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t)

cdef class ImageEditor:
    cdef:
        uint8_t[:,:,:] left_img;
        uint8_t[:,:,:] right_img;
        uint8_t[:,:,:,:] img;
    cdef unsigned int dim0,dim1,dim2,dim3,dim22,dim33

    def __cinit__(self, ndarray[uint8_t, ndim=4] img, int scale_percent):
        self.dim0 = img.shape[0]
        self.dim1 = img.shape[1]
        self.dim2 = img.shape[2]*scale_percent //100;
        self.dim3 = img.shape[3]*scale_percent//100;
        self.img = zeros((self.dim0, self.dim1, self.dim2, self.dim3), dtype=uint8)
        self.scale_img(img)
        self.left_img = self.split_img(0)
        self.right_img = self.split_img(1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=2] resizeImage(self, ndarray[uint8_t, ndim=3] imgArray):
        cdef ndarray[uint8_t, ndim=2] newSize = resize(imgArray, (self.dim2, self.dim3))
        return newSize
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=4] scale_img(self, ndarray[uint8_t, ndim=4]img):
        cdef ndarray[uint8_t, ndim=5] images = img[..., newaxis]
        cdef ndarray[uint8_t, ndim=2] image;
        #cdef  ndarray[uint8_t, ndim=3] resized;
        cdef unsigned int i, j, n, m
        for i in range(self.dim0):
            for j in range(self.dim1):
                #print("Shape images",images[i,j].shape)
                #print("Shape Buffer",asarray(self.img[i,j]).shape)
                image = self.resizeImage(images[i,j])
                #resized = image[...,newaxis]
                #print("shape of resized Img", asarray(image).shape, self.dim3)
                if(len(asarray(image).shape)>1):
                    for n in range(self.dim2):
                        for m in range(self.dim3):
                                self.img[i,j,n,m] = image[n,m]
                                #print(self.img[i,j,n,m])
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=3] split_img(self, int s):
        if(s >0):
            return asarray(self.img[:,0])
        else:
            return asarray(self.img[:,1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=3] get_left_img(self):
        return asarray(self.left_img, dtype=uint8)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=3] get_right_img(self):
        return asarray(self.right_img, dtype=uint8)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=4] get_img(self):
        return asarray(self.img, dtype=uint8)

    def images(self):
        return asarray(self.img, dtype=uint8)
    def left_image(self):
        return self.get_left_img()
    def right_image(self):
        return self.get_right_img()