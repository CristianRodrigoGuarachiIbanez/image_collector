from featureRetrieval import TrialRetriever
from groundTruthRetrieval import GroundTruthRetriever
from h5Writer import H5GTWriter, H5Writer
from buildDirector import BuildDirector
from productBuilder import ProductBuilder
from typing import Dict, List, Tuple, TypeVar, Iterator, ValuesView, Generator, Any, Callable, Union;
from numpy import ndarray, asarray, concatenate, zeros, arange, eye, max
from dataStorage import DataStorage
from cython import declare, locals, char, array, bint
from concurrent.futures import ProcessPoolExecutor, Future, as_completed

@locals(fileName=char)
def featureData(fileName: str)-> Dict[str, ndarray] :
    trial: TrialRetriever = TrialRetriever();
    # recover cvs data in trials
    return trial.callTrialDataArrAsDict(fileName);

@locals(fileName=char)
def imgData(fileName: str) ->  Dict[str, ndarray]:
    trial: TrialRetriever = TrialRetriever();
    # recover img array data as trials
    # ojo always binocular img array has to be called first in order for scene to get the data
    if(fileName =='binocular_img'):
        return trial.callImgDataArr(fileName);
    elif(fileName =='scene_img'):
        return trial.callImgDataArr(fileName);

T: TypeVar = TypeVar('T',Dict[str, int], List[str], str)

@locals(fileName=char, colRate=bint)
def labelData(filename:str, colRate: str="key") -> T:
    '''

    :param fileName: string file name
    :param colRate: sum, key
    :return: a sum of all collisions and non collisions separately
    '''
    gt: GroundTruthRetriever = GroundTruthRetriever();
    dictionary: Dict[str, int] = gt.groundTruthRetrievalOnTrial(filename, direction='la');
    if (colRate == 'sum'):
        return gt.sumValuesDict(dictionary);
    elif(colRate == 'sum2'):
        return  gt.sumValuesDictPerTrial(dictionary);
    elif(colRate == 'key'):
        return gt.getKeysDictAccordingCollision(dictionary);
    return dictionary

def saveLabelsInPickle(labelData: Callable, side: str ='la'):
    """
    this function save the GT for hand and forearm individually |lef hand, right hand | left forearm, right forearm|
    """
    executor: ProcessPoolExecutor = ProcessPoolExecutor();
    commands: List[str] = ["left_hand", "right_hand", "left_forearm", "right_forearm"]  # "left_arm", "right_arm"];
    labels: List[Dict[str, int]] = [val for val in executor.map(labelData, commands)]
    Left_hand, Right_hand, Left_forearm, Right_forearm = None, None, None, None; #type: Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]
    filename: str = None;
    if(side =='la'):
        Left_hand: Dict[str, int] = labels[0]
        print(len(Left_hand))
        Right_hand: Dict[str, int] = labels[1] #
        print(len(Right_hand))
        Left_forearm: Dict[str, int] = labels[2] #
        print(len(Left_forearm))
        Right_forearm: Dict[str, int] = labels[3] #
        print(len(Right_forearm))
        filename = 'data_left_side.txt'
    elif(side =='ra' ):
        Left_hand: Dict[str, int] = labels[0]
        print(len(Left_hand))
        Right_hand: Dict[str, int] = labels[1]  #
        print(len(Right_hand))
        Left_forearm: Dict[str, int] = labels[2]  #
        print(len(Left_forearm))
        Right_forearm: Dict[str, int] = labels[3]  #
        print(len(Right_forearm))
        filename = 'data_right_side.txt'
    # ------------------------ save ground truth data
    writer: DataStorage = DataStorage()
    labelData: List[Dict[str, int]] = [ Left_hand, Right_hand, Left_forearm, Right_forearm]
    writer.storeData(filename, data=labelData)
class LabelsReshaper:
    _data:List[ndarray]
    _data:Generator
    def __init__(self, file:Generator)->None:
        self._file = file
        self._data = list();
        self.recover_data()
    def recover_data(self)->None:
        for file in self._file:
            self._data.append(file)
    def get_data(self)->List[ndarray]:
        return self._data
    def reshape_labels(self, shape:Tuple[int,int]=(73570,6))->ndarray:
        """
        transform two arrays (N,4) in one array (N,6)
        0:HL, 1:FL, 2:HR, 3:FR, 4:BL, 5:BR
        """
        temp:ndarray = zeros(shape, dtype="int")
        for i in range(self._data[0].shape[0]):
            if(self._data[0][i,1]==1): # left hand
                temp[i,0] = 1;
            elif(self._data[0][i,2]==1): # left forearm
                temp[i,1]=1;
            elif(self._data[0][i,3]==1): # both left hand und forearm
                temp[i,4]=1;
            if(self._data[1][i,1]==1): # right hand
                temp[i,2]=1;
            elif(self._data[1][i,2]==1): # right forearm
                temp[i,3]=1;
            elif(self._data[1][i,2]==1): # both right
                temp[i,5]=1;
        assert(self._data[0].shape[0] == temp.shape[0] and self._data[0].shape[0] ==temp.shape[0]), 'shapes are not equal'
        return temp



class OneHotEncoder:
    def __init__(self, dataLabel1:Generator , dataLabel2:Generator  ) ->  None:
        self._dataLabel1: Generator = dataLabel1;
        self._dataLabel2: Generator = dataLabel2;

    def recoverLabelArraysFromGenerator(self )->ndarray: #Tuple[ndarray, ndarray]:
        """
        this convert generator with a list of arrays: [(N,), (N,)]
        """
        labels: List[array] = list();
        for gen in self._dataLabel1:
            print("lef side | right side",len(gen))
            labels.append(self.concatenateLabels(gen, next(self._dataLabel2)));
        print("labels", labels)
        print("shape generator",self.counter(labels[3]))
        def reshape(labels: List[ndarray]) -> ndarray: #Tuple[ndarray, ndarray]:
            """
            get a list of array with the length 4 -> 0.- hand left; 1.- hand right; 2.- forearm left; 3.- forearm right
            """
            assert (len(labels)>2), 'Not enough elements'
            # return self.reshapeLabelArray(labels[0], labels[2]), self.reshapeLabelArray(labels[1], labels[3]);
            return self.reshape_array(labels[0],labels[1], labels[2], labels[3])
        # return self.oneHotEncodingNumpy(reshape(labels)[0]), self.oneHotEncodingNumpy(reshape(labels)[1])
        print(reshape(labels).shape)
        return reshape(labels)
    def counter(self, dic:ndarray)->int:
        counter:int = 0
        for i in range(dic.shape[0]):
            if(dic[i]==1):
                counter+=1;
        return counter
    @staticmethod
    def reshape_array(l1:ndarray, l2:ndarray, l3:ndarray, l4:ndarray)->ndarray:
        """
        the ocurrences will be codified according to index:
        1:left hand 2: right hand 3: left forearm 4: right forearm 5: left hand and forearm 6: right hand and forearm
        return: a array (N,6)
        """
        assert(len(l1)==len(l2) and len(l2)==len(l3) and len(l3)==len(l4)), "no same length";
        new_arr:ndarray = zeros((l1.shape[0],6), dtype=int);
        for i in range(new_arr.shape[0]):
            if(l1[i]==1 and l3[i]!=1):
                new_arr[i][0] = 1;
            elif(l1[i]!=1 and l3[i]==1):
                new_arr[i][2]=1;
            elif(l2[i]==1 and l4[i]!=1):
                new_arr[i][1]=1;
            elif(l2[i]!=1 and l4[i]==1):
                new_arr[i][3]=1;
            elif(l1[i]==1 and l3[i]==1):
                new_arr[i][4]=1;
            elif(l2[i]==1 and l4[i]==1):
                new_arr[i][5]=1;
        return new_arr
    @staticmethod
    def concatenateLabels(dataLabel1: Dict[str, int], dataLabel2: Dict[str, int]) -> ndarray:
        '''
        extract the values from the dictionaries {str,int} and convert the values into arrays (N,)
        return: a concatenated array (N,)
        '''
        arrayLabel1: ndarray = asarray(list(dataLabel1.values()));
        arrayLabel2: ndarray = asarray(list(dataLabel2.values()));
        print( arrayLabel1.shape, arrayLabel2.shape)
        return concatenate((arrayLabel1, arrayLabel2));
    @staticmethod
    def reshapeLabelArray(arr1: ndarray, arr2: ndarray) -> ndarray:
        '''
        reshape every individual array from (N,) to (N,2)
        return: a array mit dim (N,2),
        '''
        assert (arr1.ndim==1 and arr2.ndim==1), 'the shape of the arrays should be (N,) with rank equal to 1'
        newShape: ndarray = zeros((arr1.shape[0],2), dtype=int);
        #print("new shape ", arr2.shape)
        assert(len(arr1) == len(arr2)), 'the length of one of the arrays is larger than the other'
        for i in range(len(arr2)):
            newShape[i,0], newShape[i,1] = arr1[i], arr2[i] # put the values of fe. left and right hand
        return newShape

    def oneHotEncodingNumpy(self, arr: ndarray) -> ndarray:
        """
        arr ist a array (N,2) 2 classes, 0 and 1
        """
        shape: Tuple[Any, Union] = (arr.shape[0], arr.shape[1]+2);
        print("One_Hotecoding",len(arr))
        one_hot: ndarray = zeros(shape);
        for i in range(len(arr)):
            if((arr[i][0]==0) and (arr[i][1]==0)):
                one_hot[i][0] = 1; # first value, no collision
            elif((arr[i][0]==1) and arr[i][1]==0):
                one_hot[i][1]=1; # # second value, collision left
            elif((arr[i][0]==0) and arr[i][1]==1):
                one_hot[i][2]=1; # third value will be 1, collision right
            elif((arr[i][0]==1) and arr[i][1]==1):
                one_hot[i][3]=1; # last value will be 1, collision both
        return one_hot

def recoverLabels(side: str) ->  List[Dict[str, int]]:

    # ---------------------------- PICKEL LOADER
    loader: DataStorage = DataStorage()
    i: int = 0;
    gen: Generator = loader.loadData(pickelFileName=side)
    # ------------------------------- recover data
    output: List[Dict[str, int]] = list();
    gtFile: Dict[str, int] = None;
    while True:
        try:
            gtFile = next(gen);
            output.append(gtFile);
        except  StopIteration as s:
            print('MAIN:', s)
            break
    return output

def main(filename: str = 'training_data.h5'):

    buildDirector: BuildDirector = BuildDirector();
    # --------------------------- WRITER
    writer: H5Writer = H5Writer(filename);
    # --------------------------- prepare image data
    biL: List[ndarray] = buildDirector.buildImgArray("binocular_img", dataToEdit='binocular_perception.h5', direction='la');
    biR: List[ndarray] = buildDirector.buildImgArray("binocular_img", dataToEdit='binocular_perception.h5', direction='ra');
    scL: List[ndarray] = buildDirector.buildImgArray('scene_img', dataToEdit='scene_records.h5', direction='la')
    scR: List[ndarray] = buildDirector.buildImgArray('scene_img', dataToEdit='scene_records.h5', direction='ra')

    # ------------------------------- prepare labels data
    labelLeftSide: List[Dict[str, int]] = recoverLabels('data_left_side')
    labelRightSide: List[Dict[str, int]] = recoverLabels('data_right_side')

    data: List[Any] = [biL, biR, scL, scR]
    for i in range(len(data)):
        if(i == 0 and len(data[i])> 0):
            data  = [data[i], labelLeftSide[0], labelLeftSide[1], labelLeftSide[2], labelLeftSide[3]]
            datasetnames: List[str] = ['binocular_features_left', 'gt_hl_left_hand', 'gt_hl_right_hand', 'gt_hl_right_forearm', 'gt_hl_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='binocular_left_side', datasetNames=datasetnames)
        elif(i ==1 and len(biR)>0):
            data = [data[i], labelRightSide[0], labelRightSide[1], labelRightSide[2], labelRightSide[3]]
            datasetnames: List[str] = ['binocular_features_right', 'gt_hr_left_hand', 'gt_hr_right_hand', 'gt_hr_right_forearm', 'gt_hr_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='binocular_right_side', datasetNames=datasetnames)
        elif(i ==2 and len(scL)>0):
            data = [data[i], labelLeftSide[0], labelLeftSide[1], labelLeftSide[2], labelLeftSide[3]]
            datasetnames: List[str] = ['scene_features_left', 'gt_hl_left_hand', 'gt_hl_right_hand', 'gt_hl_right_forearm', 'gt_hl_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='scene_left_side', datasetNames=datasetnames)
        elif (i ==3 and len(scR) > 0):
            data = [data[i], labelRightSide[0], labelRightSide[1], labelRightSide[2], labelRightSide[3]]
            datasetnames: List[str] = ['scene_features_right', 'gt_hr_left_hand', 'gt_hr_right_hand',  'gt_hr_right_forearm', 'gt_hr_right_forearm']
            writer.saveImgDataIntoGroup(imgData=data, groupName='scene_right_side', datasetNames=datasetnames)

    writer.closingH5PY()

if __name__ == '__main__':

    # executor: ProcessPoolExecutor = ProcessPoolExecutor();
    # commands: List[str] = ["left_hand", "right_hand", "left_forearm", "right_forearm"] #"left_arm", "right_arm"];
    # # featureCommands: List[str] = ["left_hand", "right_hand", "left_forearm", "right_forearm", "left_arm", "right_arm", "joint_coord", "object_coord", "head_coord", "hand_coord"]
    #
    # ----------------------------- retrieve Labels

    # # labels: List[Future] = [executor.submit(labelData,  command, 'sum2') for command in commands]
    # # features: List[Future] = [executor.submit(featureData, featureCommand) for featureCommand in featureCommands ]
    # # features: List[Dict[str, ndarray]] = [val for val in executor.map(featureData, featureCommands)]
    # # print(len(features))
    # print(len(labels))

    # ----------------------- COLLISIONS

    # left_hand:  Dict[str, ndarray] = features[0]#featureData('left_hand');
    # right_hand: Dict[str, ndarray] = features[1]#featureData('right_hand');
    # left_forearm: Dict[str, ndarray] = features[2]#featureData('left_forearm');
    # right_forearm: Dict[str, ndarray] = features[3]#featureData('right_forearm');
    # left_arm: Dict[str, ndarray] = features[4]
    # right_arm: Dict[str, ndarray] = features[5]
    #
    # head_coord: Dict[str, ndarray] = features[8]
    # hand_coord: Dict[str, ndarray] = features[9]
    # object_coord: Dict[str, ndarray] = features[7]
    # joints: Dict[str, ndarray] = features[6] #featureData('joint_coord');

    # --------------------- LABELS

    # print(len(joints));
    # print('LEFT HAND', len(left_hand))
    # print('RIGHT HAND', len(right_hand))
    # print('LEFT FOREARM', len(left_forearm))
    # print('RIGHT FOREARM',len(right_forearm))
    # print('RIGHT ARM',len(right_arm))
    # print('LEFT ARM', len(left_arm))
    #
    # print('HEAD COORD', len(head_coord))
    # print('HAND COORD', len(hand_coord))
    # print('JOINTS', len(joints))
    # print('OBJECT COORD', len(object_coord))

    # print('LEFT HAND', len(gt_hl_Right_hand))
    # print('RIGHT HAND', len(gt_hl_Right_hand))
    # print('LEFT FOREARM',len(gt_hl_Left_forearm))
    # print('RIGHT FOREARM',len(gt_hl_Right_forearm))
    # print('LEFT HAND', len(gt_hr_Right_hand))
    # print('RIGHT HAND', len(gt_hr_Right_hand))
    # print('LEFT FOREARM', len(gt_hr_Left_forearm))
    # print('RIGHT FOREARM', len(gt_hr_Right_forearm))
    #main()
    # saveLabelsInPickle(labelData=labelData, side='la')
    p = DataStorage()
    gen: Generator = p.loadData('data_left_side.txt')#
    gen2: Generator = p.loadData('data_right_side.txt')
    onehotencoding: OneHotEncoder = OneHotEncoder(gen, gen2)
    concatenatedArrays:ndarray = onehotencoding.recoverLabelArraysFromGenerator();
    #print(concatenatedArrays[0].shape)
    #print(concatenatedArrays[1].shape)
    counter:int=0
    for i in range(concatenatedArrays.shape[0]):
        if(concatenatedArrays[i][1]==1):
            counter+=1
    print(counter)

    writer: DataStorage = DataStorage()
    #writer.storeData('label_data_6C.txt', concatenatedArrays)

    file:Generator = p.loadData('label_data.txt')
    data: LabelsReshaper = LabelsReshaper(file)
    print(data.get_data()[0].shape)
    new_labels:ndarray =data.reshape_labels()
    print(new_labels.shape)
    writer.storeData('label_data_6C.txt', new_labels)



