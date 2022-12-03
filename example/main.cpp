#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ELSED.h"

cv::Mat CV_Imread1920x1080(const std::string& file)
{
  cv::Mat img = cv::imread(file);
  cv::resize(img, img, cv::Size(1440, 1080), 0, 0, cv::INTER_NEAREST);
  cv::resize(img, img, cv::Size(960, 720), 0, 0, cv::INTER_NEAREST);
  //cv::resize(img, img, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
  //cv::resize(img, img, cv::Size(256, 192), 0, 0, cv::INTER_NEAREST);
  return img;
}

inline void
drawSegments(cv::Mat img,
             upm::Segments segs,
             const cv::Scalar &color,
             int thickness = 1,
             int lineType = cv::LINE_AA,
             int shift = 0)
{
  for (const upm::Segment &seg: segs)
    cv::line(img, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), color, thickness, lineType, shift);
}

int main() {
  std::cout << "******************************************************" << std::endl;
  std::cout << "******************* ELSED main demo ******************" << std::endl;
  std::cout << "******************************************************" << std::endl;

  // Using default parameters (long segments)
  cv::Mat img = CV_Imread1920x1080("../images/55_scene.png");
  if (img.empty()) {
    std::cerr << "Error reading input image" << std::endl;
    return -1;
  }

  upm::ELSED elsed;
  upm::Segments segs = elsed.detect(img);
  std::cout << "ELSED detected: " << segs.size() << " (large) segments" << std::endl;

  drawSegments(img, segs, CV_RGB(0, 255, 0), 2);
  cv::imshow("ELSED long", img);
  cv::waitKey();

  // Not using jumps (short segments)
  img = CV_Imread1920x1080("../images/55_scene.png");
  if (img.empty()) {
    std::cerr << "Error reading input image" << std::endl;
    return -1;
  }

  upm::ELSEDParams params;
  params.listJunctionSizes = {};
  upm::ELSED elsed_short(params);
  segs = elsed_short.detect(img);
  std::cout << "ELSED detected: " << segs.size() << " (short) segments" << std::endl;

  drawSegments(img, segs, CV_RGB(0, 255, 0), 2);
  cv::imshow("ELSED short", img);
  cv::waitKey();

  return 0;
}