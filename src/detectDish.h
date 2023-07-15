//
// Created by eb0r0h on 10/06/23.
//

#ifndef FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISH_H
#define FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISH_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>




cv::Mat detectDishesEdge(cv::Mat image);

cv::Mat watershedByOpencCV(cv::Mat src);

cv::Mat augmentation( cv::Mat image, float factor);

cv::Mat segmentationHope(cv::Mat dishes0);

cv::Mat detectSalad(cv::Mat image);

cv::Mat detectSaladBowl(const cv::Mat& image);

cv::Mat segmentImage(const cv::Mat& inputImage, int k);

//
cv::Mat removeDishes(cv::Mat image, int delta);



#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISH_H
