import cython
from numpy cimport ndarray, int_t, uint8_t
from numpy import asarray, newaxis, resize, zeros, uint8
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t)

cdef enum LeftV:
        xls = 142# 126
        xle = 462 #478
        yls = 135 #121
        yle = 375 #387

cdef enum RightV:
    xrs = 566 #550
    xre = 886 #901
    yrs = 135 #121
    yre = 375 #387
cdef class IMG_CROPPER:

    cdef:
        uint8_t[:,:] resizedLeftImg
        uint8_t[:,:] resizedRightImg
        uint8_t[:,:] left_img
        uint8_t[:,:] right_img
        LeftV left_values
        RightV right_values
    def __cinit__(self, ndarray[uint8_t, ndim=2] img):
        self.cropLeftImg(img)
        self.cropRightImg(img)
        self.resizedLeftImg = self.resizeImg(side=0, scale=50)
        self.resizedRightImg = self.resizeImg(side=1, scale=50)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cropLeftImg(self,ndarray[uint8_t, ndim=2] img):
        #print(yls)
        self.left_img = img[yls:yle, xls:xle]
        #print(asarray(self.left_img).shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cropRightImg(self,ndarray[uint8_t, ndim=2] img):
        self.right_img = img[yrs:yre, xrs:xre]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=2] resizeImg(self, int side, int scale ):
        cdef uint8_t[:,:] img;
        if(side ==0):
            img = self.left_img
        else:
            img = self.right_img

        cdef int w = img.shape[1]*scale //100;
        cdef int h = img.shape[0]*scale//100;
        return resize(img, (w,h))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=2] get_left_img(self):
        return asarray(self.left_img, dtype=uint8)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=2] get_right_img(self):
        return asarray(self.right_img, dtype=uint8)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=2] getResizedLeftImg(self):
        return asarray(self.resizedLeftImg, dtype=uint8)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint8_t, ndim=2] getResizedRightImg(self):
        return asarray(self.resizedRightImg, dtype=uint8)

    def resized_left_img(self):
        return self.getResizedLeftImg()
    def resized_right_img(self):
        return self.getResizedRightImg()

    def left_image(self):
        return self.get_left_img()
    def right_image(self):
        return self.get_right_img()