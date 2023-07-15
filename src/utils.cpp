//
// Created by eb0r0h on 15/07/23.
//


#include "utils.h"
using namespace std;
using namespace cv;


vector<Mat> createVecImgFromSource(string path)
{
    vector<Mat> images;
    vector<string> fileNames;
    glob(path, fileNames);
    for (const auto& filename : fileNames)
    {
        Mat image = imread(filename);
        images.push_back(image);
    }
    return images;

}

