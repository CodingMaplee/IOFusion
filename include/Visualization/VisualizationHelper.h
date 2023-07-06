//
// Created by user on 2022/9/14.
//
#ifndef MAINLOOP_VISUALIZATIONHELPER_H
#define MAINLOOP_VISUALIZATIONHELPER_H

#pragma once
#include <iostream>
#include <string>

class VisualizationHelper{
public:
    static void ShowUint(uint* mask, uint width, uint height, std::string windowName)
    {
        cv::Mat geometryMask = cv::Mat(height, width, CV_8UC3);
        for(int i = 0;i<height;i++)
        {
            for(int j = 0;j<width;j++)
            {
                if(mask[j + i * width] == 1)
                {
                    geometryMask.at<cv::Vec3b>(i, j)[0] = 255;
                    geometryMask.at<cv::Vec3b>(i, j)[1] = 255;
                    geometryMask.at<cv::Vec3b>(i, j)[2] = 255;
                }
                else
                {
                    geometryMask.at<cv::Vec3b>(i, j)[0] = 0;
                    geometryMask.at<cv::Vec3b>(i, j)[1] = 0;
                    geometryMask.at<cv::Vec3b>(i, j)[2] = 0;
                }
            }
        }
        //cv::imshow(windowName, geometryMask);
        std::string name =windowName+".png";
        cv::imwrite (name,geometryMask);
        cv::waitKey(1);
    }

};




#endif //MAINLOOP_VISUALIZATIONHELPER_H
