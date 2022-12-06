//
// Created by zxm on 2022/12/5.
//

#include <cfenv>
#include <string>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Tools.h"


cv::Mat zxm::CV_Convert32FTo32FC3(const cv::Mat& m, const cv::Mat& mask)
{
  CV_Assert(m.type() == CV_32F);
  CV_Assert(mask.type() == CV_8U);
  cv::Mat o(m.size[0], m.size[1], CV_32FC3);
  for (int i=0; i<m.size[0]; ++i)
    for (int j=0; j<m.size[1]; ++j)
    {
      if (mask.at<uint8_t>(i, j) <= 1)
      {
        o.at<cv::Vec3f>(i, j) = {1, 0, 0};//指向照片内部
      } else
      {
        o.at<cv::Vec3f>(i, j)[0] = m.at<float>(i, j, 0);
        o.at<cv::Vec3f>(i, j)[1] = m.at<float>(i, j, 1);
        o.at<cv::Vec3f>(i, j)[2] = m.at<float>(i, j, 2);
        o.at<cv::Vec3f>(i, j) = cv::normalize(o.at<cv::Vec3f>(i, j));
      }
    }
  return o;
}


cv::Mat zxm::CV_Resize256x192(const cv::Mat& m, int interpolation)
{
  CV_Assert(!m.empty());
  cv::Mat o;
  cv::resize(m, o, cv::Size(480, 360), 0, 0, interpolation);
  cv::resize(o, o, cv::Size(256, 192), 0, 0, interpolation);
  return o;
}


cv::Mat zxm::CV_Imread1920x1440(const std::string& file, int imreadFlag, int interpolation)
{
  cv::Mat img = cv::imread(file, imreadFlag);
  CV_Assert(!img.empty());
  cv::resize(img, img, cv::Size(1440, 1080), 0, 0, interpolation);
  //cv::resize(img, img, cv::Size(960, 720), 0, 0, interpolation);
  return img;
}


void zxm::SampleAColor(double *color, double x, double min, double max)
{
  /*
   * Red = 0
   * Green = 1
   * Blue = 2
   */
  double posSlope = (max-min)/60;
  double negSlope = (min-max)/60;

  if( x < 60 )
  {
    color[0] = max;
    color[1] = posSlope*x+min;
    color[2] = min;
    return;
  }
  else if ( x < 120 )
  {
    color[0] = negSlope*x+2*max+min;
    color[1] = max;
    color[2] = min;
    return;
  }
  else if ( x < 180  )
  {
    color[0] = min;
    color[1] = max;
    color[2] = posSlope*x-2*max+min;
    return;
  }
  else if ( x < 240  )
  {
    color[0] = min;
    color[1] = negSlope*x+4*max+min;
    color[2] = max;
    return;
  }
  else if ( x < 300  )
  {
    color[0] = posSlope*x-4*max+min;
    color[1] = min;
    color[2] = max;
    return;
  }
  else
  {
    color[0] = max;
    color[1] = min;
    color[2] = negSlope*x+6*max;
    return;
  }
}


void zxm::CV_ImWriteWithPath(const std::string &path, const cv::Mat &im)
{
  namespace fs = std::filesystem;
  fs::path file(path);
  auto parentPath = file.parent_path();
  if (!fs::exists(parentPath))
    fs::create_directories(parentPath);
  cv::imwrite(path, im);
}


cv::Mat zxm::DrawClusters(const std::string &savePath,
                          const cv::Mat &clustersMap)
{
  CV_Assert(clustersMap.type()==CV_32S);
  const int Rows = clustersMap.rows, Cols = clustersMap.cols;
  const int32_t MinC = -1, MaxC = 1 + cvMax<int32_t>(clustersMap);
  //扰乱c(0~MaxC)，让颜色分布更乱，过滤掉-1（不指向任何颜色）
  std::vector<int32_t> colorBar(MaxC + 1);
  std::iota(colorBar.begin(), colorBar.end(), 0);
  shuffle(colorBar);
  cv::Mat colorMap(Rows, Cols, CV_8UC3, cv::Scalar_<uint8_t>(0, 0, 0));
  for (size_t i=0; i<Rows; ++i)
    for (size_t j=0; j<Cols; ++j)
    {
      int32_t c = clustersMap.at<int32_t>((int)i, (int)j);
      if (c < 0)
        continue;
      c = colorBar[c];
      double color[3];
      SampleAColor(color, 360.*(double)c/(double)colorBar.size(), 0, 255);
      // OpenCV color is BGR.
      colorMap.at<cv::Vec3b>((int)i, (int)j)[0] = (uint8_t)(color[2]);
      colorMap.at<cv::Vec3b>((int)i, (int)j)[1] = (uint8_t)(color[1]);
      colorMap.at<cv::Vec3b>((int)i, (int)j)[2] = (uint8_t)(color[0]);
    }
  CV_ImWriteWithPath(savePath, colorMap);
  return colorMap;
}


cv::Mat zxm::DrawNormals(const std::string &savePath, const cv::Mat &normals)
{
  CV_Assert(normals.type()==CV_32FC3);
  const int Rows = normals.size[0], Cols = normals.size[1];
  cv::Mat _normals = normals.clone();
  cv::Mat colorMap(Rows, Cols, CV_8UC3, cv::Scalar_<uint8_t>(0, 0, 0));
  for (size_t i=0; i<Rows; ++i)
    for (size_t j=0; j<Cols; ++j)
    {
      const cv::Vec3f &normal = _normals.at<cv::Vec3f>((int)i, (int)j);
      double color[3] = {normal[0], normal[1], normal[2]};
      color[0] = clamp((color[0] / 2 + 0.5) * 255, 0., 255.);
      color[1] = clamp((color[1] / 2 + 0.5) * 255, 0., 255.);
      color[2] = clamp((color[2] / 2 + 0.5) * 255, 0., 255.);
      // OpenCV color is BGR.
      colorMap.at<cv::Vec3b>((int)i, (int)j)[0] = (uint8_t)(color[2]);
      colorMap.at<cv::Vec3b>((int)i, (int)j)[1] = (uint8_t)(color[1]);
      colorMap.at<cv::Vec3b>((int)i, (int)j)[2] = (uint8_t)(color[0]);
    }
  CV_ImWriteWithPath(savePath, colorMap);
  return colorMap;
}

void zxm::CheckMathError() {
  std::ostringstream oss;
  if constexpr (math_errhandling | MATH_ERREXCEPT) {
    int err = std::fetestexcept(FE_ALL_EXCEPT);
    if (err == 0)
      return;
    if (err & FE_INEXACT) {
      std::feclearexcept(FE_ALL_EXCEPT);
      //忽视精度丢失的异常
      return;
    }
    oss << "Math Error:\n";
    if (err & FE_INVALID)
      oss << "Domain error FE_INVALID\n";
    if (err & FE_DIVBYZERO)
      oss << "Pole error FE_DIVBYZERO\n";
    if (err & FE_OVERFLOW)
      oss << "Range error due to overflow FE_OVERFLOW\n";
    if (err & FE_UNDERFLOW)
      oss << "Range error due to underflow FE_UNDERFLOW\n";
    std::feclearexcept(FE_ALL_EXCEPT);
    throw std::runtime_error(oss.str());
  } else if constexpr (math_errhandling | MATH_ERRNO) {
    if (errno == 0)
      return;
    oss
    << "Math Error:\n"
    << std::strerror(errno);
    throw std::runtime_error(oss.str());
  }//else assert(0).
}

