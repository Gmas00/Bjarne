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

/*
void filterAreas(const cv::Mat& input,Mat& output,int threshold)
{

    //Mat gray;
    //cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    findContours(input, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++)
    {
        if (contourArea(contours[i]) > threshold)
        {
            drawContours(output, contours, i, 255, -1);
        }
    }
}*/

