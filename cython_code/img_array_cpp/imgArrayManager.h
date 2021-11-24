#define IMG_ARRAY_CPP_IMGARRAYMANAGER_H

#include <zip.h>
#include<string>
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <boost/filesystem.hpp>
using std::vector;
using std::string;
namespace fs = boost::filesystem;

class ImgArrayManager{
    private:
        vector<fs::path> dir;
        vector<fs::path>paths:
    public:
    ImgArrayManager(fs::path const &root, string const & ext ){
        dir = getDirectories(root, ext);
    }
    vector<fs::path> getDirectories(fs::path const &root, string const & ext){
        if (fs::exists(root) && fs::is_directory(root)){

            for (auto const & entry : fs::recursive_directory_iterator(root))
            {
                if (fs::is_regular_file(entry) && entry.path().extension() == ext)
                    paths.emplace_back(entry.path().filename());
            }
        }
        else{
         std::cout<< " this is no directory or does not exist"<<std::endl;
         }
        return paths;
        }
    vector<fs::path> getZipFiles(string filename);
    vector<ImgArrayContainer> getImgArrays(string imgfilename);
};

class ImgArrayContainer{
    vector<cv::Mat> container;
    void setImgArray(cv::Mat img);
    cv::Mat getImgArrays();
}
