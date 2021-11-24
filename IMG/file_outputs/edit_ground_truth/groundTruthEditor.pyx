from numpy cimport ndarray, int32_t
from numpy import concatenate, asarray, zeros, int32
from libc.string cimport memset
from libc.stdlib cimport malloc, free
cdef class GTEditor:
    cdef:
        int32_t[:] gt_hand, gt_farm
        int length
        int** gt
        int32_t[:] left_hand, right_hand, left_farm, right_farm
    def __cinit__(self, int32_t[:] left_hand, int32_t[:] right_hand, int32_t[:] left_farm, int32_t[:] right_farm):
        self.left_hand = left_hand
        self.left_farm = left_farm
        self.right_hand = right_hand
        self.right_farm = right_farm
        assert(self.checkLength()), "the length of the arrays is not equal"
        assert(self.checkDim()), "the dimensions of the arrays are not equal"
        self.length = left_hand.shape[0]
        self.gt =  <int**>malloc(left_hand.shape[0]*sizeof(int*))
        self.populate()
        self.fillUp()
        if(self.gt ==NULL):
            raise MemoryError()
    def __deallocate__(self):
        free(self.gt)
    cdef int checkDim(self):
        if(len(self.left_hand.shape)==len(self.right_hand.shape)):
            if(len(self.left_hand.shape)==len(self.left_farm.shape)):
                if(len(self.left_farm.shape)==len(self.right_farm.shape)):
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    cdef int checkLength(self):
        if(self.left_hand.shape[0]==self.right_hand.shape[0]):
            if(self.left_hand.shape[0]==self.left_farm.shape[0]):
                if(self.left_farm.shape[0]==self.right_farm.shape[0]):
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    cdef void populate(self):
        cdef int i
        for i in range(self.length):
            self.gt[i] = <int*> malloc(6*sizeof(int))
    cdef void fillUp(self):
        cdef int i
        for i in range(self.length):
            if(self.left_hand[i]==1 and self.left_farm[i]==0):
                self.gt[i][0] = 1
            elif(self.right_hand[i]==1 and self.right_farm[i]==0):
                self.gt[i][1]=1
            elif(self.left_farm[i]==1 and self.left_hand[i]==0):
                self.gt[i][2]=1
            elif(self.right_farm[i]==1 and self.left_hand[i]==0):
                self.gt[i][3]=1
            elif(self.left_hand[i]==1 and self.left_farm[i]==1):
                self.gt[i][4]=1
            elif(self.right_hand[i]==1 and self.left_farm[i]==1):
                self.gt[i][5]=1
    cdef list[list[int]] convertGTtoPython(self, int**gt):
        cdef:
            unsigned int i,j
        matrix = []

        for i in range(self.length):
            submatrix = []
            for j in range(6):
                submatrix.append(gt[i][j])
            matrix.append(submatrix)
        return matrix

    def getGTArray(self):
        output = asarray(self.convertGTtoPython(self.gt))
        return output
