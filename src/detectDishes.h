//
// Created by Gabriel on 10/06/23.
//

#ifndef FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISHES_H
#define FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISHES_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



cv::Mat detectSalad(const cv::Mat& image);

cv::Mat detectDishesEdge(const cv::Mat& image);

cv::Mat detectFoods(const cv::Mat& image);

cv::Mat detectBread(const cv::Mat& image);



//tentativi
cv::Mat watershedByOpencCV(cv::Mat src);

cv::Mat augmentation( cv::Mat image, float factor);

cv::Mat segmentationHope(cv::Mat dishes0);

cv::Mat segmentImage(const cv::Mat& inputImage, int k);

//ste
cv::Mat removeDishes(cv::Mat image, int delta);



#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISHES_H
