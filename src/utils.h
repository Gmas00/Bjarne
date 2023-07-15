//
// Created by eb0r0h on 15/07/23.
//

#ifndef FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_UTILS_H
#define FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_UTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>


std::vector<cv::Mat> createVecImgFromSource(std::string path);
cv::Mat filterAreas(const cv::Mat& input,int threshold);


#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_UTILS_H
