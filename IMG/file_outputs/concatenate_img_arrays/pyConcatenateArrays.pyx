from numpy cimport ndarray, uint8_t
from numpy import ones, zeros, asarray, uint8
import cython
ctypedef unsigned char uchar
cdef class PyConcatArray:
    cdef:
        unsigned int rows
        unsigned int cols
        unsigned int length
        uchar[:,:,:] data

    def __cinit__(self, ndarray[uchar, ndim=3] d,  ndarray[uchar, ndim=3] dt):
        self.length = d.shape[0] + dt.shape[0];
        self.rows = d.shape[1]
        self.cols = dt.shape[2]
        self.data = zeros((self.length,self.rows,self.cols), dtype=uint8)
        self.concatenate_arrays(d, 0, d.shape[0])
        self.concatenate_arrays2(dt, d.shape[0],self.length)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void concatenate_arrays(self, uchar[:,:,:] img, unsigned int start, unsigned int end):
        cdef:
            unsigned int i, j, k
        print(self.length, self.rows, self.cols, start)
        for i in range(start, end):
            for j in range(self.rows):
                for k in range(self.cols):
                    #print(i,j,k)
                    self.data[i,j,k] = img[i,j,k]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void concatenate_arrays2(self, uchar[:,:,:] img, int start, int end):
        cdef:
            int j , k
            int counter = 0
        while(start<end):
            for j in range(self.rows):
                for k in range(self.cols):
                    #print(start, j, k)
                    self.data[start,j,k] = img[counter,j,k]
            counter+=1
            start+=1
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef uint8_t[:,:,:] getConcatenatedImg(self):
        return asarray(self.data, dtype=uint8)

    def getData(self):
        return self.getConcatenatedImg()