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
cv::Mat watershedByOpencCV(cv::Mat src);

std::vector<cv::Mat> createVecImgFromSource(std::string path);

cv::Mat augmentation( cv::Mat image, float factor);

cv::Mat segmentationHope(cv::Mat dishes0);

cv::Mat detectSalad(cv::Mat image);



#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTFOOD_H
