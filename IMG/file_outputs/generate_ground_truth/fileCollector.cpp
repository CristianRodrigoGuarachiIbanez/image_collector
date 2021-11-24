

#include<zip.h>
#include <iostream>

using std::cout
using std::endl;
struct zip_stat st;
class GTG{
public:
    int err;
    zip *z;
    char*zipname
    char*filename;
    char *contents;
    zip_file * file;
    GTG(const char *zipname, const char * filename){
        this->err=0;
        this-> z = zip_open(zipname, 0, &err);
    }
    ~GTG(){
        cout<<"Destructor"<<endl;
    }
    extractCSVFile(const char*filename){
        // iterate for all csv files

        zip_start_init(&st);
        zip_stat(z,filename, 0, &st);
        //allocates memory
        contents = new char[st.size];

    }


};