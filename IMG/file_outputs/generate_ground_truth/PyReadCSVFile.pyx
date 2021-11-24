from numpy cimport ndarray, int32_t
from numpy import ones, zeros, asarray, int32
cdef class PyReadCSVFiles:
    cdef:
        unsigned int rows
        unsigned int cols
        double[:,:] data
        int32_t[:] gt
    def __cinit__(self, ndarray[double, ndim=2] data):
        self.data = data;
        self.rows = data.shape[0]
        self.cols = data.shape[1]
        self.gt = zeros((self.rows,), dtype=int32)
        self.generateGT()
    cdef void generateGT(self):
        cdef:
            unsigned int i, j

        for i in range(self.rows):
            for j in range(self.cols):
                if(self.data[i,j]>0):
                    self.gt[i]=1;
                else:
                    pass
    cdef int32_t[:] getGT(self):
        return self.gt
    def getData(self):
        return asarray(self.getGT())