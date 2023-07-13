//
// Created by eb0r0h on 10/06/23.
//

#ifndef FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTFOOD_H
#define FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTFOOD_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat detectDishEdge(cv::Mat image);
cv::Mat testolo(cv::Mat image);

std::vector<cv::Mat> createVecImgFromSource(std::string path);




#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTFOOD_H
