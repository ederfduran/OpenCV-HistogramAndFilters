#ifndef BUTTONS_H
#define BUTTONS_H

// OpenCV includes
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <string>
#include <vector>

namespace btn_cbk
{

cv::Mat img;

void showHistoCallback(int state, void *)
{
    //Separate image channels in BRG
    std::vector<cv::Mat> bgr;
    // get the channels
    cv::split(img, bgr);
    // Create the histogram for 256 bins
    // The number of possibles values [0..255] number of bins
    int numbins = 256;
    // set the ranges for BGR last is not included
    float range[] = {0, 256};
    const float *hits_range = range;
    // to store each histogram
    cv::Mat b_hist, g_hist, r_hist;
    // *** CALCULATE EACH CHANNEL HISTOGRAM **
    // input img , num of images, channels dimensions,optional mask matrix, variable to store calculated histogram,
    // Histogram dimensionality, num of bins to calculate(one for each pixel), range of input variables
    cv::calcHist(&bgr[0], 1, 0, cv::Mat(), b_hist, 1, &numbins, &hits_range);
    cv::calcHist(&bgr[1], 1, 0, cv::Mat(), g_hist, 1, &numbins, &hits_range);
    cv::calcHist(&bgr[2], 1, 0, cv::Mat(), r_hist, 1, &numbins, &hits_range);

    // Draw the histogram
    // We go to draw lines for each channel
    int width = 512;
    int height = 300;
    // Create image with gray base
    cv::Mat hist_image(height, width, CV_8UC3, cv::Scalar(20, 20, 20));

    // Normalize the histograms to height of image
    cv::normalize(b_hist, b_hist, 0, height, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, height, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, height, cv::NORM_MINMAX);

    int bin_step = cvRound((float)width / (float)numbins);
    for (int i = 1; i < numbins; i++)
    {
        cv::line(hist_image,
                cv::Point(bin_step * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
                cv::Point(bin_step * (i), height - cvRound(b_hist.at<float>(i))),
                cv::Scalar(255, 0, 0));
        cv::line(hist_image,
                cv::Point(bin_step * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
                cv::Point(bin_step * (i), height - cvRound(g_hist.at<float>(i))),
                cv::Scalar(0, 255, 0));
        cv::line(hist_image,
                cv::Point(bin_step * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
                cv::Point(bin_step * (i), height - cvRound(r_hist.at<float>(i))),
                cv::Scalar(0, 0, 255));
    }

    imshow("Histogram", hist_image);
}

// Increase the contrast of an image
void equalizeCallback(int state, void *)
{
    cv::Mat result;
    // Convert BGR image to YCbCr,YCbCr is not an absolute color space,instead is a way to codify RGB information.
    // Y represents the luminance component . Cb and Cr are the crominance difference between blue and red .
    // To equalize a color image, we only have to equalize the luminance channel
    cv::Mat ycrcb;
    // BGR -> YCbCr
    cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);

    // Split image into channels
    std::vector<cv::Mat> channels;
    split(ycrcb, channels);

    // Equalize the Y channel only in,out
    equalizeHist(channels[0], channels[0]);

    // Merge the result channels
    merge(channels, ycrcb);

    // Convert ycrcb -> BGR
    cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);

    // Show image
    cv::imshow("Equalized", result);
}

void lomoCallback(int state, void *)
{
    cv::Mat result;

    const double exponential_e = std::exp(1.0); // Euler
    // create look-up table(rows, cols, type) for color curve effect
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++)
    {
        float x = (float)i / 256.0;
        /**
         *      
         *      _____1_______
         *             - x_-0.5__
         *      1 +  e      s   
         */
        lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(exponential_e, -((x - 0.5) / 0.1)))));
    }
    // split the image channels and apply curve transform only to red channel
    std::vector<cv::Mat> bgr;
    cv::split(img, bgr);
    //  input img, look up table , output img
    // apply the curve only for red channel
    cv::LUT(bgr[2], lut, bgr[2]); 
    cv::merge(bgr, result);

    // Create image for halo dark               Gray Image
    cv::Mat halo(img.rows, img.cols, CV_32FC3, cv::Scalar(0.3, 0.3, 0.3));
    // Create circle inside
    cv::circle(halo, cv::Point(img.cols / 2, img.rows / 2), img.cols / 3, cv::Scalar(1, 1, 1), -1);
    cv::blur(halo, halo, cv::Size(img.cols / 3, img.cols / 3));

    // Convert the result to a float to allow multiply  by 1 factor
    cv::Mat resultf;
    result.convertTo(resultf, CV_32FC3);

    // multiply our result with halo
    cv::multiply(resultf, halo, resultf);

    //convert to 8 bit
    resultf.convertTo(result, CV_8UC3);

    //show result
    cv::imshow("Lomograpy", result);
}

void edgeProcess(cv::Mat &img_cannyf)
{
    /** EDGES **/
    // Apply median filter to remove possible noise / Gaussian blur work aswell
    cv::Mat img_median;
    //         input img, out img, kernel size.
    //a kernel is a small matrix used to apply some mathematical operation, such as convolutional means, to an image
    //Blurs an image using the median filter.
    cv::medianBlur(img, img_median, 7);

    // Detect edges with canny algorithm
    cv::Mat img_canny;
    // input img, out img , first threshold, second threshold
    //The function finds edges in the input image and marks them in the output map edges using the Canny algorithm.
    cv::Canny(img_median, img_canny, 50, 150);

    // Dilate the edges to join broken edges
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    //         input img, out img , kernel img
    cv::dilate(img_canny, img_canny, kernel);

    // Scale edges values to 1 and invert values
    img_canny = img_canny / 255;
    img_canny = 1 - img_canny;

    // Use float values to allow multiply between 0 and 1
    img_canny.convertTo(img_cannyf, CV_32FC3);

    // Blur the edgest to do smooth effect
    cv::blur(img_cannyf, img_cannyf, cv::Size(5, 5));
}

void cartoonCallback(int state, void *)
{

    cv::Mat img_cannyf;
    edgeProcess(img_cannyf);
    /** COLOR **/
    // Apply bilateral filter to homogenizes color
    // The bilateral filter is a filter that reduces the noise of an image while keeping the edges
    cv::Mat img_bf;
    // in img, out img, Diameter of pixel neighborhood, sigma color value, sigma coordinate space
    // With a diameter greater than five, the bilateral filter starts to become slow. With sigma values greater than 150, a cartoonish effect appears.
    bilateralFilter(img, img_bf, 9, 150.0, 150.0);

    // truncate colors to stronger catoonish effect
    cv::Mat result = img_bf / 25;
    result = result * 25;

    /** MERGES COLOR + EDGES **/
    // Create a 3 channles for edges
    cv::Mat img_canny3c;
    cv::Mat cannyChannels[] = {img_cannyf, img_cannyf, img_cannyf};
    merge(cannyChannels, 3, img_canny3c);

    // Convert color result to float
    cv::Mat resultf;
    result.convertTo(resultf, CV_32FC3);

    // Multiply color and edges matrices
    cv::multiply(resultf, img_canny3c, resultf);

    // convert to 8 bits color
    resultf.convertTo(result, CV_8UC3);

    // Show image
    cv::imshow("Result", result);
}

} // namespace btn_cbk

#endif