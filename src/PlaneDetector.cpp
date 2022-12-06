//
// Created by zxm on 2022/12/5.
//


#include <vector>
#include <opencv2/core.hpp>

#include "ELSED.h"
#include "../HyperConfig.h"
#include "PlaneDetector.h"
#include "Tools.h"


cv::Mat zxm::DetectPlanes(const cv::Mat &colorImg,
                          const cv::Mat &normalMap,
                          bool enableDebug)
{
  cv::Mat clusterMap;
  zxm::ClusteringByNormal(clusterMap, normalMap);
  cv::Mat clusterColorMap;
  if (enableDebug)
    clusterColorMap = DrawClusters("../dbg/NormalClusters.png", clusterMap);
  auto lineMap = zxm::CreateStructureLinesMap(colorImg, normalMap);
  if (enableDebug)
    DrawClusters("../dbg/Lines.png", lineMap);
  if (enableDebug)
  {
    const int Rows = clusterColorMap.size[0], Cols = clusterColorMap.size[1];
    CV_Assert(Rows==lineMap.size[0] && Cols==lineMap.size[1]);
    for (int i = 0; i < Rows; ++i)
      for (int j = 0; j < Cols; ++j)
        if (lineMap.at<int32_t>(i, j) > 0)
          clusterColorMap.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
    CV_ImWriteWithPath("../dbg/BlendNormalClustersAndLines.png", clusterColorMap);
  }
  zxm::SegmentByLines(clusterMap, lineMap, enableDebug);
  return clusterMap;
}


int zxm::ClusteringByNormal(cv::Mat &clusterMap, const cv::Mat &normalMap)
{
  using Pxi32_t = std::pair<int,int>;
  CV_Assert(normalMap.type()==CV_32FC3);

  const int Rows = normalMap.size[0], Cols = normalMap.size[1];
  auto isValidIndex = [&Rows,&Cols](int y, int x)->bool
  {
    return 0 <= y && y < Rows && 0 <= x && x < Cols;
  };
  // first pass
  ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t& yxOff :{Pxi32_t{0,-1},
                                  Pxi32_t{-1,-1},
                                  Pxi32_t{-1,0},
                                  Pxi32_t{-1,1}})
      {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (isValidIndex(targetY, targetX))
        {
          //fixme: 阈值
          constexpr float thContinuedAngle = 10.f;
          bool isContinued = false;
          const cv::Vec3f v1 = normalMap.at<cv::Vec3f>(targetY, targetX);
          /*
          const cv::Vec3f v0 =
            isValidIndex(targetY+yxOff.first, targetX+yxOff.second) && !isOnLine(targetY,targetX) ?
            normalMap.at<cv::Vec3f>(targetY + yxOff.first, targetX + yxOff.second) : v1;
          */
          const cv::Vec3f v2 = normalMap.at<cv::Vec3f>(i, j);
          /*
          const cv::Vec3f v3 =
            isValidIndex(i - yxOff.first, j - yxOff.second) && !isOnLine(i, j) ?
            normalMap.at<cv::Vec3f>(i - yxOff.first, j - yxOff.second) : v2;
          */
          isContinued = acos(clamp(v1.dot(v2), -1.f, 1.f)) <= (thContinuedAngle * CV_PI / 180);
          zxm::CheckMathError();
          if (isContinued)
          {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            if (C < 0)
              throw std::logic_error("Processed label of (targetY,targetX) should >= 0!"
                                     " in zxm::ClusteringByNormal()");
            connectedCls.emplace(C);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }//检测(i,j)与周围像素的连通性
      if (connectedCls.empty())
      {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      }
      else
      {
        if (connectedCls.size() > 1)
          for(int C : connectedCls)
            if (C != minConnectedCls)
              mergeCls.unionClass(minConnectedCls, C);
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass over.
  std::vector<uint32_t> shrinkClass;
  int nCls = (int)mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      int32_t C = resultCls.at<int32_t>(i, j);
      C = (int32_t)shrinkClass[C];
      resultCls.at<int32_t>(i, j) = C;
    }
  // resultCls...
  swap(resultCls, clusterMap);
  return nCls;
}


cv::Mat zxm::CreateStructureLinesMap(const cv::Mat &colorImg,
                                     const cv::Mat &normalMap)
{
  CV_Assert(colorImg.type()==CV_8UC3);
  CV_Assert(normalMap.type()==CV_32FC3);
  CV_Assert(colorImg.size[0] >= normalMap.size[0]);
  CV_Assert(colorImg.size[1] >= normalMap.size[1]);

  int Rows = colorImg.size[0], Cols = colorImg.size[1];
  upm::ELSED lineDetector;
  upm::ImageEdges edges = lineDetector.detectEdges(colorImg);
  if (Rows > normalMap.size[0] && Cols > normalMap.size[1])
  {
    //线段检测结果放缩到normalMap大小
    double
    scaleY = (double)normalMap.size[0] / Rows,
    scaleX = (double)normalMap.size[1] / Cols;
    Rows = normalMap.size[0];
    Cols = normalMap.size[1];
    for (auto& e : edges)
      for (auto& px : e)
      {
        px.y = int((px.y + 0.5) * scaleY);
        px.x = int((px.x + 0.5) * scaleX);
      }
  }
  //判断线段是纹理线还是结构线（两侧深度or深度不一致）：用法向量代理深度检测结构线会有一点问题。
  auto isDiffSide = [&normalMap](int iy0, int ix0, int iy1, int ix1)
  {
    const auto
    &v0 = normalMap.at<cv::Vec3f>(iy0, ix0),
    &v1 = normalMap.at<cv::Vec3f>(iy1, ix1);
    const float angle = acos(clamp(v0.dot(v1), -1.f, 1.f));
    zxm::CheckMathError();
    //fixme：阈值
    return angle >= float(20 * CV_PI / 180);
  };
  //绘制边缘热图result
  cv::Mat result(Rows, Cols, CV_32S, cv::Scalar_<uint8_t>(0));
  int32_t startID = 1;
  for (const auto& e : edges)
  {
    if (e.size() < 2)
      continue;
    //计算每个像素应该向两侧偏移的距离(y0,x0)和(y1,x1)
    const float
    deltaY = float(e.back().y - e.front().y),
    deltaX = float(e.back().x - e.front().x);
    //fixme：半径阈值
    const float Radius = 1.5f;
    const float
    dy = (deltaY / sqrt(deltaY * deltaY + deltaX * deltaX)) * Radius,
    dx = (deltaX / sqrt(deltaY * deltaY + deltaX * deltaX)) * Radius;
    const float
    y0 = -dx, x0 = dy,
    y1 = dx, x1 = -dy;//(dy,dx) 分别逆时针、顺时针转动90°
    zxm::CheckMathError();
    //检测边e两侧的法向量是否一致
    size_t nDiffSide = 0;
    for (const auto &px: e)
    {
      int iy0 = int(px.y + 0.5f + y0),
          ix0 = int(px.x + 0.5f + x0),
          iy1 = int(px.y + 0.5f + y1),
          ix1 = int(px.x + 0.5f + x1);
      if (0 <= iy0 && iy0 < Rows && 0 <= ix0 && ix0 < Cols &&
          0 <= iy1 && iy1 < Rows && 0 <= ix1 && ix1 < Cols &&
          isDiffSide(iy0, ix0, iy1, ix1))
        ++nDiffSide;
    }
    //fixme: 阈值
    if (nDiffSide >= size_t(0.45f * e.size()))
    {
      //如果边缘e上有足够多的像素左右不一致，则表明e是一条结构线而不是纹理线
      for (const auto &px: e)
        result.at<int32_t>(px.y, px.x) = startID;
      ++startID;
    }
  }
  return result;
}


int zxm::SegmentByLines(cv::Mat &clusterMap,
                        const cv::Mat &lineMap,
                        bool enableDebug)
{
  using Pxi32_t = std::pair<int,int>;
  CV_Assert(clusterMap.type() == CV_32S);
  CV_Assert(lineMap.type() == CV_32S);
  CV_Assert(clusterMap.size == lineMap.size);

  const int Rows = clusterMap.size[0], Cols = clusterMap.size[1];
  // first pass
  ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      // 计算当前像素(i,j)和周围4领域像素的连通性
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t& yxOff :{Pxi32_t{0,-1}, Pxi32_t{-1,0}})
      {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols)
        {
          int32_t
          cTarget = clusterMap.at<int32_t>(targetY, targetX),
          cCur    = clusterMap.at<int32_t>(i, j);
          int32_t
          targetLine = lineMap.at<int32_t>(targetY, targetX),
          curLine    = lineMap.at<int32_t>(i, j);
          /*
           * 连通性检测：
           * t在线上，c也在线上：聚类ID相等；
           * t在线上，c不在：   一定不联通；
           * t不在线上，c在：   聚类ID相等；
           * t不在线上，c也不在：聚类ID相等；
           */
          if (!(targetLine>0 && curLine<=0) && (cTarget==cCur))
          {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            if (C < 0)
              throw std::logic_error("Processed label of (targetY,targetX) should >= 0!"
                                     " in zxm::SegmentByLines()");
            connectedCls.emplace(C);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }
      if (connectedCls.empty())
      {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      }
      else
      {
        if (connectedCls.size() > 1)
          for(int C : connectedCls)
            if (C != minConnectedCls)
            {
              mergeCls.unionClass(minConnectedCls, C);
            }
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass for all pixels

  if (enableDebug)
    zxm::DrawClusters("../dbg/RawLineClusters.png", resultCls);

  std::vector<uint32_t> shrinkClass;
  int nCls = (int)mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i=0; i<Rows; ++i)
    for (int j=0; j<Cols; ++j)
    {
      int32_t C = resultCls.at<int32_t>(i, j);
      resultCls.at<int32_t>(i, j) = (int)shrinkClass[C];
    }
  swap(clusterMap, resultCls);
  return nCls;
}
