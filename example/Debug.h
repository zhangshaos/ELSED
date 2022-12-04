//
// Created by zxm on 2022/12/3.
//

#ifndef ELSED_DEBUG_H
#define ELSED_DEBUG_H


#include <string>
#include <filesystem>
#include <execution>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>


template<typename T>
T max(const cv::Mat& m)
{
  T maxV = std::numeric_limits<T>::min();
  std::for_each(std::execution::seq,
                m.begin<T>(),
                m.end<T>(),
                [&maxV](const T& v)
                {
                  if (v > maxV)
                    maxV = v;
                });
  return maxV;
}


inline void
SampleAColor(double *color, double x, double min, double max)
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


inline void
ImWriteWithPath(const std::string &path, const cv::Mat &im)
{
  namespace fs = std::filesystem;
  fs::path file(path);
  auto parentPath = file.parent_path();
  if (!fs::exists(parentPath))
    fs::create_directories(parentPath);
  cv::imwrite(path, im);
}


inline void
DrawClusters(const std::string &savePath,
             const cv::Mat &clustersMap)
{
  assert(clustersMap.type()==CV_32S);
  const int Rows = clustersMap.rows, Cols = clustersMap.cols;
  const int32_t MinC = -1, MaxC = 1 + max<int32_t>(clustersMap);

  cv::Mat colorMap(Rows, Cols, CV_8UC3, cv::Scalar_<uint8_t>(0, 0, 0));
  for (size_t i=0; i<Rows; ++i)
    for (size_t j=0; j<Cols; ++j)
    {
      int32_t c = clustersMap.at<int32_t>((int)i, (int)j);
      if (c < 0)
        continue;
      double color[3];
      SampleAColor(color, 360.*(double)(c-MinC)/(double)(MaxC-MinC), 0, 255);
      // OpenCV color is BGR.
      colorMap.at<cv::Vec3b>((int)i, (int)j)[0] = (uint8_t)(color[2]);
      colorMap.at<cv::Vec3b>((int)i, (int)j)[1] = (uint8_t)(color[1]);
      colorMap.at<cv::Vec3b>((int)i, (int)j)[2] = (uint8_t)(color[0]);
    }
  ImWriteWithPath(savePath, colorMap);
}


template<typename T>
inline
T clamp(T v, T minV, T maxV)
{
  return v < minV ? minV : (v > maxV ? maxV : v);
}


inline void
DrawNormals(const std::string &savePath, const cv::Mat &normals)
{
  assert(normals.type()==CV_32FC3);
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
  ImWriteWithPath(savePath, colorMap);
}

#endif //ELSED_DEBUG_H
