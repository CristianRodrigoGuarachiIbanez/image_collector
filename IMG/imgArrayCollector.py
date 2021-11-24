
from zipfile import ZipFile
from typing import List, Type
from os import listdir, getcwd, remove, mkdir
from os.path import isfile, join, exists
from h5py import File
from pickle import dump
from cv2 import imread, IMREAD_GRAYSCALE, resize, INTER_NEAREST,  cvtColor, COLOR_BGR2GRAY, imshow, waitKey
from h5Writer import H5Writer
from img import ImageEditor
from crop_img.crop_img import IMG_CROPPER
from imageEditor import  ImageEditor as IE
from file_outputs.concatenate_img_arrays.pyConcatenate import PyConcatArray
from file_outputs.edit_ground_truth.groundTruthEditor import GTEditor
from file_outputs.generate_ground_truth.groundTruth import PyReadCSVFiles
from numpy import ndarray, asarray, uint8, newaxis, genfromtxt, concatenate, double
from pandas import read_csv
class IMG(object):
    _zipfile:ZipFile;
    _path:str;
    left_imgs:List[ndarray];
    right_imgs:List[ndarray];
    gt_left_hand:ndarray
    gt_right_hand:ndarray
    gt_left_farm:ndarray
    gt_right_farm:ndarray
    def __init__(self, path:str):
        self._zipfile = None
        self._currPath = getcwd()
        self.left_imgs = list()
        self.right_imgs = list()
        self.left_gaze_left_img = list()
        self.left_gaze_right_img = list()
        self.right_gaze_left_img = list()
        self.right_gaze_right_img = list()

        self.right_gaze_left_hand = list()
        self.right_gaze_right_hand = list()
        self.left_gaze_left_hand = list()
        self.left_gaze_right_hand = list()

        self.right_gaze_left_farm = list()
        self.right_gaze_right_farm = list()
        self.left_gaze_left_farm = list()
        self.left_gaze_right_farm = list()

        self.__selectDirectories(path)

    def __selectDirectories(self, directory:str):
        dir:str;
        inputB:str = "./binocular_image_data.h5"
        inputS:str = "./scene_image_data.h5"
        writer = H5Writer(inputB)
        datasets: List[str] = ["left_img", "right_img"]
        dataset: str = "scene_img"
        if(directory):
            dir = directory
        else:
            dir = self._currPath

        for file in listdir(dir):
            if (file.endswith('.zip')):
                globalPath: str = self._currPath +"/trials/" + file
                print("global path", globalPath)
                try:
                    self.__extractFromZipFile(globalPath, archive="image_outputs");
                except Exception as e:
                    print('[Error]: IMG ARRAY :', e);

        #print(asarray(self.left_gaze_left_img).shape, asarray(self.right_gaze_left_img).shape)
        #print(asarray(self.left_gaze_right_img).shape, asarray(self.right_gaze_right_img).shape)
        left_imgs = PyConcatArray(asarray(self.left_gaze_left_img, dtype=uint8), asarray(self.right_gaze_left_img, dtype=uint8))
        right_imgs = PyConcatArray(asarray(self.left_gaze_right_img, dtype=uint8), asarray(self.right_gaze_right_img, dtype=uint8))
        imgL = left_imgs.getData()
        imgR = right_imgs.getData()
        print("length of left imgs:", imgL.shape)
        print("length of right imgs:", imgR.shape)

        imgData: List[ndarray] = [asarray(imgL, dtype=uint8), asarray(imgR, dtype=uint8)]
        writer.saveImgDataIntoGroup(imgData, "binocular_image", datasetNames=datasets)

        assert (len(self.left_gaze_left_hand)==len(self.left_gaze_right_farm)), "the length of left hand and forearm are not equal"
        assert (len(self.right_gaze_right_hand)) ==len(self.right_gaze_left_hand),"the length of left hand and forearm are not equal"

        self.gt_left_hand = concatenate((self.__concatenateInList(gaze_side="left_left_hand"), self.__concatenateInList(gaze_side="right_left_hand")), axis=None)
        self.gt_right_hand = concatenate((self.__concatenateInList(gaze_side="left_right_hand"), self.__concatenateInList(gaze_side="right_right_hand")), axis=None)
        self.gt_left_farm =  concatenate((self.__concatenateInList(gaze_side="left_left_farm"), self.__concatenateInList(gaze_side="right_left_farm")), axis=None)
        self.gt_right_farm = concatenate((self.__concatenateInList(gaze_side="left_right_farm"), self.__concatenateInList(gaze_side="right_right_farm")), axis=None)
        print("GT right Farm",self.gt_right_farm.shape)
        print("GT left Farm",self.gt_left_farm.shape)
        print("GT right hand",self.gt_right_hand.shape)
        print("GT Left Hand",self.gt_left_hand.shape)
        GT = GTEditor(self.gt_left_hand, self.gt_right_hand, self.gt_left_farm, self.gt_right_farm)
        gt_final = GT.getGTArray()
        print("FINAL SHAPE:",gt_final.shape)
        openPickle= open("./label_data.txt", "ab")
        dump(gt_final, openPickle)
        openPickle.close()
    def __extractFromZipFile(self, zipFileName:str, archive:str) -> None:
        '''
        :param zipFileName: string;
        :param archive: string name of the file to read
        '''
        fuzzRatio:float;
        img:ndarray;
        self._zipfile = ZipFile(zipFileName, "r")
        zipFiles: List[str] = self._zipfile.namelist();  # names of the files in the Zip
        filename = zipFileName.split("/")[-1]
        for i in range(len(zipFiles)):
            if(zipFiles[i].endswith(".png") and zipFiles[i].startswith(archive)):
                if(zipFiles[i].endswith("_None.png")):
                    print("excluded",zipFiles[i])
                    continue
                else:

                    img = self.__openImgFiles(zipFiles[i], dataset="binocularPerception")
                    #print("SHAPE BEFORE:", img.shape)
                    cropped_img = IMG_CROPPER(img)
                    crop_right_img = cropped_img.right_image()
                    #self.right_imgs.append(self.__resizeImg(crop_right_img, scale=50))
                    #print("TYPE", crop_right_img.dtype)
                    crop_left_img = cropped_img.left_image()
                    #self.left_imgs.append(self.__resizeImg(crop_left_img, scale=50))
                    if (filename.startswith("la_") and filename.endswith(".zip")):  # indicates the gaze direction
                        self.left_gaze_right_img.append(self.__resizeImg(crop_right_img, scale=50))
                        self.left_gaze_left_img.append(self.__resizeImg(crop_left_img, scale=50))
                    elif (filename.startswith("ra_") and filename.endswith(".zip")):
                        self.right_gaze_left_img.append(self.__resizeImg(crop_left_img, scale=50) )
                        self.right_gaze_right_img.append(self.__resizeImg(crop_right_img, scale=50))
            elif(zipFiles[i].endswith(".csv") and zipFiles[i].startswith("forearm_left")): # indicates the hand side
                file = self.__extractCSVFiles(zipFiles[i])
                gt_farm = PyReadCSVFiles(file)
                gt = gt_farm.getData()
                print("arm files shape", gt.shape)
                if (filename.startswith("la_") and filename.endswith(".zip")): # indicates the gaze direction
                    self.left_gaze_left_farm.append(gt)
                elif (filename.startswith("ra_") and filename.endswith(".zip")):
                    self.right_gaze_left_farm.append(gt)
            elif (zipFiles[i].endswith(".csv") and zipFiles[i].startswith("forearm_right")):
                file = self.__extractCSVFiles(zipFiles[i])
                gt_farm = PyReadCSVFiles(file)
                gt = gt_farm.getData()
                print("arm files shape", gt.shape)
                if (filename.startswith("la_") and filename.endswith(".zip")):
                    self.left_gaze_right_farm.append(gt)
                elif (filename.startswith("ra_") and filename.endswith(".zip")):
                    self.right_gaze_right_farm.append(gt)
            elif(zipFiles[i].endswith(".csv") and zipFiles[i].startswith("hand_left")):
                file = self.__extractCSVFiles(zipFiles[i])
                gt_hand= PyReadCSVFiles(file)
                gt = gt_hand.getData()
                print("hand files shape", gt.shape)
                if (filename.startswith("la_") and filename.endswith(".zip")):
                    self.left_gaze_left_hand.append(gt)
                elif (filename.startswith("ra_") and filename.endswith(".zip")):
                    self.right_gaze_left_hand.append(gt)
            elif (zipFiles[i].endswith(".csv") and zipFiles[i].startswith("hand_right")):
                file = self.__extractCSVFiles(zipFiles[i])
                gt_hand = PyReadCSVFiles(file)
                gt = gt_hand.getData()
                print("hand files shape", gt.shape)
                if (filename.startswith("la_") and filename.endswith(".zip")):
                    self.left_gaze_right_hand.append(gt)
                elif (filename.startswith("ra_") and filename.endswith(".zip")):
                    self.right_gaze_right_hand.append(gt)
    @staticmethod
    def __resizeImg(img:ndarray, scale:int)->ndarray:
        w:int = img.shape[1]*scale//100;
        h:int = img.shape[0]*scale//100;
        return resize(img, (w,h))
    @staticmethod
    def __addChannel(img:ndarray)->ndarray:
        return img[..., newaxis]
    def __concatenateInList(self, gaze_side:str)->ndarray:
        output:List[ndarray];
        if(gaze_side == "left_left_hand"):
            output = self.left_gaze_left_hand
        elif(gaze_side=="left_right_hand"):
            output = self.left_gaze_right_hand
        elif(gaze_side=="left_left_farm"):
            output = self.left_gaze_left_farm
        elif(gaze_side=="left_right_farm"):
            output = self.left_gaze_right_farm
        elif(gaze_side=="right_right_hand"):
            output = self.right_gaze_right_hand
        elif(gaze_side=="right_left_hand"):
            output = self.right_gaze_left_hand
        elif(gaze_side =="right_left_farm"):
            output = self.right_gaze_left_farm
        elif(gaze_side=="right_right_farm"):
            output = self.right_gaze_right_farm
        assert(len(output)>0), "the list of arrays is empty"
        arr1 = output[0]
        for i in range(1,len(output)):
            arr1 =  concatenate((arr1, output[i]), axis=None)
        return arr1
    def __openH5file(self, filename:str, dataset:str)->ndarray:
        imgArray: ndarray = None;
        with File(filename, "r") as file:
            print("...those are the keys/datasets from inside of the class", file.keys())
            try:
                imgArray = asarray(file.get(dataset));
                print("Size of List", len(imgArray), "Shape:", imgArray.shape)

            except Exception as e:
                print(e)
                print("the data set {} could not be opened".format(filename));
            file.close()
            return imgArray
    def __openImgFiles(self, filename:str, dataset:str):
        filePath: str = self._currPath + "/" + filename
        self._zipfile.extract(filename, self._currPath)
        img = imread(filePath, IMREAD_GRAYSCALE);
        if (isfile(filePath)):
            remove(filePath)
        return asarray(img)

    def __extractCSVFiles(self, filename:str):
        filePath: str = self._currPath + "/" + filename
        self._zipfile.extract(filename, self._currPath)
        file = genfromtxt(filename, delimiter=",")
        # file = read_csv(filename, header=None, skiprows=[1])
        # file = file.to_numpy(dtype=double)
        file = file[1:,1:]
        # print("file numpy array",file.shape)
        if(isfile(filePath)):
            remove(filePath)
        return file

if __name__ == "__main__":
    img = IMG("./trials")
    # from imgArrayRecoverer import ImgArrayRecoverer
    #
    # img = imread("./image_outputs/binocular_view_316.png", IMREAD_GRAYSCALE);
    # i = ImgArrayRecoverer()
    # i.binocularSetter(img)
