# distutils: language = c++
from numpy cimport ndarray, ndarray, int_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from numpy cimport ndarray, uint8_t
from numpy import asarray
cdef class ImgArrayRecoverer:
    cdef:
     list[ndarray] left_imgs;
     list[ndarray] right_imgs;
     ndarray[uint8_t, ndarray=2] left_img;
     ndarray[uint8_t, ndarray=2] right_img;

    def __cinit__(self):
        self.left_imgs = list();
        self.right_imgs = list()
    cdef ndarray[int_t, ndim=2] splitter(self, ndarray[int_t, ndim=4] img, int left):
        if(left > 1):
             return img[0,:,:,0]
         else:
            return img[1,:,:,0]
    cdef void binocularSetter(self, ndarray[uint8_t, ndim=4] img):
        try:
            left_img = self.splitter(img, 1)
            right_img = self.splitter(img,0)
        except Exception as e:
            print("[Info]", e)

        self.right_imgs.append(right_img)
        self.left_imgs.append(left_img)

    cdef ndarray[int_t, ndim=4] binocularGetter(self, int side):
    assert(len(self.right_img) == len(self.right_img)), "the size of left and right images are not similar"
        if(side == 1):
            return asarray(self.right_imgs)
         else:
            return asarray(self.left_imgs)





