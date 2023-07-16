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
//Function that given an image, returns a new image with only the salad in the picture
cv::Mat detectSalad(const cv::Mat& image);
//Function that given an image, returns a new image with only the dishes in the picture
cv::Mat detectDishesEdge(const cv::Mat& image);
//Function that given an image, returns a new image with dishes and salad in the picture
cv::Mat detectFoods(const cv::Mat& image);
//Function that given an image, returns a new image with only the bread in the picture
cv::Mat detectBread(const cv::Mat& image);


//ste
cv::Mat removeDishes(cv::Mat image, int delta);




#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISHES_H
