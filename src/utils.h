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
//void filterAreas(const cv::Mat& input,cv::Mat& output,int threshold);

std::vector<std::string> getLabels();


bool compareColors(const std::pair<cv::Vec3b, int>& color1, const std::pair<cv::Vec3b, int>& color2);

std::vector<cv::Vec3b> findMostFrequentColors(const cv::Mat& image, int numColors);

cv::Mat removeColors(const cv::Mat& image1,int size,int numColors,int delta);

#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_UTILS_H
