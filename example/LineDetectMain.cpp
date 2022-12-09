//
// Created by zxm on 2022/12/8.
//
// 经过测试，发现ELSED的线段检测数目，没有EDLines高，
// 不同照片下，两者的效果各有千秋，因此最好结合法向量结构线检测来一决雌雄。


#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ELSED.h"


inline cv::Mat
CV_Imread1920x1080(const std::string& file)
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
  std::cout << "**************** Line Detect main demo ***************" << std::endl;
  std::cout << "******************************************************" << std::endl;

  // Using default parameters (long segments)
  cv::Mat img = CV_Imread1920x1080("../images/55_scene.png");
  if (img.empty()) {
    std::cerr << "Error reading input image" << std::endl;
    return -1;
  }

  upm::ELSEDParams elsedParams;
  elsedParams.anchorThreshold = 4;
  elsedParams.gradientThreshold = 20;
  elsedParams.ksize = 3;
  elsedParams.sigma = 1.2;
  elsedParams.minLineLen = 9;
  //elsedParams.listJunctionSizes = {5,7,9}; 不好调整
  upm::ELSED elsed(elsedParams);
  upm::Segments segs = elsed.detect(img);
  std::cout << "ELSED detected: " << segs.size() << " segments" << std::endl;

  //将图片和边缘放缩到256x192
  double scaleY = 192. / elsed.getImgInfo().imageHeight,
         scaleX = 256. / elsed.getImgInfo().imageWidth;
  for (auto &seg : segs) {
    // x0, y0, x1, y1
    seg(0) *= scaleX;
    seg(1) *= scaleY;
    seg(2) *= scaleX;
    seg(3) *= scaleY;
  }
  cv::resize(img, img, cv::Size(480, 360), 0, 0, cv::INTER_NEAREST);
  cv::resize(img, img, cv::Size(256, 192), 0, 0, cv::INTER_NEAREST);

  drawSegments(img, segs, CV_RGB(0, 255, 0), 2);
  cv::imwrite("ELSED result.jpg", img);

  //保存边缘图
  cv::Mat result(192, 256, CV_8U, cv::Scalar_<uint8_t>(0));
  for (const auto &seg : segs) {
    int
    ix0 = int(seg(0)),
    iy0 = int(seg(1)),
    ix1 = int(seg(2)),
    iy1 = int(seg(3));
    cv::line(result, {ix0, iy0}, {ix1, iy1}, cv::Scalar_<uint8_t>(255));
  }
  cv::imwrite("ELSED edge.jpg", result);
  //cv::waitKey();
  return 0;
}