# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool, string

cdef extern from "<algorithm>" namespace "std":
    Iter find_if[Iter, Func](Iter first, Iter last, Func pred);

cdef bool findOne(int elem):
    if elem ==1:
        return True;

cdef char getNonCollisionsOnly(dict dictData):
    cdef unsigned int noncollisions, collisions, value;
    cdef char previousKey, currKey;
    cdef vector[int] tempList;
    cdef vector[char] keys
    cdef vector[int].iterator found
    cdef char key
    for key, value in dictData.items():
        keys = key.split(',');
        currKey= keys[1];
        if(currKey == previousKey):
            tempList.push_back(value)
        else:
            previousKey = currKey;
            if(tempList.size() != 0):
                found = find_if(tempList.begin(), tempList.end(), findOne);
                if(found != tempList.end()):
                   collisions += 1;  # if 1 in tempList else 0;
                else:
                   noncollisions += 1,   # if not 1 in tempList else 0;

            tempList.clear()
            tempList.push_back(value)

    return 'number of frames {} | number of trials {} | non-collisions:{} | collisions: {}'.format(len(dictData), noncollisions + collisions, noncollisions, collisions);