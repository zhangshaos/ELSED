//
// Created by zxm on 2022/12/5.
//


#include <map>
#include <vector>
#include <opencv2/core.hpp>

#include "ELSED.h"
#include "../HyperConfig.h"
#include "PlaneDetector.h"
#include "Tools.h"


cv::Mat zxm::DetectPlanes(const cv::Mat &colorImg,
                          const cv::Mat &normalMap,
                          bool enableDebug) {
  cv::Mat clusterMap;
  zxm::ClusteringByNormal(clusterMap, normalMap, true);
  cv::Mat clusterColorMap;
  if (enableDebug)
    clusterColorMap = DrawClusters("../dbg/NormalClusters.png", clusterMap);
  auto lineMap = zxm::CreateStructureLinesMap(colorImg, normalMap);
  if (enableDebug)
    DrawClusters("../dbg/Lines.png", lineMap);
  if (enableDebug) {
    const int Rows = clusterColorMap.size[0], Cols = clusterColorMap.size[1];
    CV_Assert(Rows == lineMap.size[0] && Cols == lineMap.size[1]);
    for (int i = 0; i < Rows; ++i)
      for (int j = 0; j < Cols; ++j)
        if (lineMap.at<int32_t>(i, j) > 0)
          clusterColorMap.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
    CV_ImWriteWithPath("../dbg/BlendNormalClustersAndLines.png", clusterColorMap);
  }
  zxm::SegmentByLines(clusterMap, lineMap, enableDebug);
  return clusterMap;
}


int zxm::ClusteringByNormal(cv::Mat &clusterMap,
                            const cv::Mat &normalMap,
                            bool enableDebug) {
  using Pxi32_t = std::pair<int, int>;
  CV_Assert(normalMap.type() == CV_32FC3);

  const int Rows = normalMap.size[0], Cols = normalMap.size[1];
  auto isValidIndex = [&Rows, &Cols](int y, int x) -> bool {
    return 0 <= y && y < Rows && 0 <= x && x < Cols;
  };
  // first pass
  ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j) {
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1,-1},
                                  Pxi32_t{-1, 0},
                                  Pxi32_t{-1, 1}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (isValidIndex(targetY, targetX)) {
          bool isContinued = false;
          const cv::Vec3f &v1 = normalMap.at<cv::Vec3f>(targetY, targetX);
          /*
          const cv::Vec3f v0 =
            isValidIndex(targetY+yxOff.first, targetX+yxOff.second) && !isOnLine(targetY,targetX) ?
            normalMap.at<cv::Vec3f>(targetY + yxOff.first, targetX + yxOff.second) : v1;
          */
          const cv::Vec3f &v2 = normalMap.at<cv::Vec3f>(i, j);
          /*
          const cv::Vec3f v3 = isValidIndex(i - yxOff.first, j - yxOff.second) ?
            normalMap.at<cv::Vec3f>(i - yxOff.first, j - yxOff.second) : v2;
          */
          isContinued = acos(clamp(v1.dot(v2), -1.f, 1.f)) <= (TH_CONTINUED_ANGLE * CV_PI / 180);
          zxm::CheckMathError();
          if (isContinued) {
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
      if (connectedCls.empty()) {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      } else {
        if (connectedCls.size() > 1)
          for (int C: connectedCls)
            if (C != minConnectedCls)
              mergeCls.unionClass(minConnectedCls, C);
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass over.

  if (enableDebug)
    zxm::DrawClusters("../dbg/RawNormalClusters.png", resultCls);

  std::vector<uint32_t> shrinkClass;
  int nCls = (int) mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j) {
      int32_t C = resultCls.at<int32_t>(i, j);
      C = (int32_t) shrinkClass[C];
      resultCls.at<int32_t>(i, j) = C;
    }
  // resultCls...
  swap(resultCls, clusterMap);
  return nCls;
}


cv::Mat zxm::CreateStructureLinesMap(const cv::Mat &colorImg,
                                     const cv::Mat &normalMap) {
  CV_Assert(colorImg.type() == CV_8UC3);
  CV_Assert(normalMap.type() == CV_32FC3);
  CV_Assert(colorImg.size[0] >= normalMap.size[0]);
  CV_Assert(colorImg.size[1] >= normalMap.size[1]);

  int Rows = colorImg.size[0], Cols = colorImg.size[1];
  upm::ELSED lineDetector;
  upm::ImageEdges edges = lineDetector.detectEdges(colorImg);
  //todo：更好的划线算法，而不是直接放缩
  if (Rows > normalMap.size[0] && Cols > normalMap.size[1]) {
    //线段检测结果放缩到normalMap大小
    double
      scaleY = (double) normalMap.size[0] / Rows,
      scaleX = (double) normalMap.size[1] / Cols;
    Rows = normalMap.size[0];
    Cols = normalMap.size[1];
    for (auto &e: edges)
      for (auto &px: e) {
        px.y = int((px.y + 0.5) * scaleY);
        px.x = int((px.x + 0.5) * scaleX);
      }
  }
  //判断线段是纹理线还是结构线（两侧深度or深度不一致）：用法向量代理深度检测结构线会有一点问题。
  auto isDiffSide = [&normalMap](int iy0, int ix0, int iy1, int ix1) {
    const auto
      &v0 = normalMap.at<cv::Vec3f>(iy0, ix0),
      &v1 = normalMap.at<cv::Vec3f>(iy1, ix1);
    const float angle = acos(clamp(v0.dot(v1), -1.f, 1.f));
    zxm::CheckMathError();
    return angle >= float(TH_DIFF_SIDE_ANGLE * CV_PI / 180);
  };
  //绘制边缘热图result
  cv::Mat result(Rows, Cols, CV_32S, cv::Scalar_<int32_t>(-1));
  int32_t startID = 1;
  for (const auto &e: edges) {
    if (e.size() < 2)
      continue;
    //计算每个像素应该向两侧偏移的距离(y0,x0)和(y1,x1)
    const float
      deltaY = float(e.back().y - e.front().y),
      deltaX = float(e.back().x - e.front().x);
    const float
      dy = (deltaY / sqrt(deltaY * deltaY + deltaX * deltaX)) * GAP_HALF_DIFF_SIDE,
      dx = (deltaX / sqrt(deltaY * deltaY + deltaX * deltaX)) * GAP_HALF_DIFF_SIDE;
    const float
      y0 = -dx, x0 = dy,
      y1 = dx, x1 = -dy;//(dy,dx) 分别逆时针、顺时针转动90°
    zxm::CheckMathError();
    //检测边e两侧的法向量是否一致
    size_t nDiffSide = 0;
    for (const auto &px: e) {
      int iy0 = int(px.y + 0.5f + y0),
        ix0 = int(px.x + 0.5f + x0),
        iy1 = int(px.y + 0.5f + y1),
        ix1 = int(px.x + 0.5f + x1);
      if (0 <= iy0 && iy0 < Rows && 0 <= ix0 && ix0 < Cols &&
          0 <= iy1 && iy1 < Rows && 0 <= ix1 && ix1 < Cols &&
          isDiffSide(iy0, ix0, iy1, ix1))
        ++nDiffSide;
    }
    if (nDiffSide >= size_t(STRUCTURE_LINE_RATIO * e.size())) {
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
                        bool enableDebug) {
  using Pxi32_t = std::pair<int, int>;
  CV_Assert(clusterMap.type() == CV_32S);
  CV_Assert(lineMap.type() == CV_32S);
  CV_Assert(clusterMap.size == lineMap.size);

  const int Rows = clusterMap.size[0], Cols = clusterMap.size[1];
  // first pass
  ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j) {
      int32_t cCur = clusterMap.at<int32_t>(i, j);
      int32_t curLine = lineMap.at<int32_t>(i, j);
      if (curLine > 0)
        //线段上像素点的归属稍后再判断
        continue;
      //计算当前像素(i,j)和周围4领域像素的连通性
      std::set<int> connectedCls;//与当前像素联通的像素，他们的类别应该被合并
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          //解决穿透效应，对在线上的像素单独处理。
          //1、c和t都不在线上： 若类ID相等，则联通，并更新connectedCls和maxConnectedCls
          //2、c不在线上，t在： 不联通
          if (targetLine <= 0 && (cTarget == cCur)) {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            if (C < 0)
              throw std::logic_error("Processed label of (targetY,targetX) should >= 0!"
                                     " in zxm::SegmentByLines()");
            connectedCls.emplace(C);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }//for all adjacent pixel
      if (connectedCls.empty()) {
        //新类别
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      } else {
        if (connectedCls.size() > 1)
          for (int C: connectedCls)
            if (C != minConnectedCls)
              mergeCls.unionClass(minConnectedCls, C);
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass for all pixels
  //单独对边界上的像素点处理，赋予类别
  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j) {
      int32_t cCur = clusterMap.at<int32_t>(i, j);
      int32_t curLine = lineMap.at<int32_t>(i, j);
      if (curLine <= 0)
        //只处理线段上的像素归属问题
        continue;
      //1、如果当前像素周围（4邻域）有cID一致的不在线上的项，则和其一致；
      //2、不满足1时，如果周围存在之前处理的cID一致且也在线上的项，则和其一致；
      //3、不满足2时，周围存在还未处理的cID一致且也在线上的项，暂时处理不了，创建新ID吧
      //4、否则（cID不一致），创建新ID。
      bool found = false;
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0},
                                  Pxi32_t{0,  1},
                                  Pxi32_t{1,  0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          if (cTarget==cCur && targetLine<=0) {
            resultCls.at<int32_t>(i, j) = resultCls.at<int32_t>(targetY, targetX);
            found = true;//break do-while
            break;//break for
          }
        }
      }
      if (found)
        continue;//go to next line pixel
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          if (cTarget==cCur && targetLine>0) {
            resultCls.at<int32_t>(i, j) = resultCls.at<int32_t>(targetY, targetX);
            found = true;//break do-while
            break;//break for
          }
        }
      }
      if (found)
        continue;//go to next line pixel
      //否则，创建新ID
      mergeCls.tryInsertClass(nextCls);
      resultCls.at<int32_t>(i, j) = nextCls++;
    }//one pass for line pixel

  if (enableDebug)
    zxm::DrawClusters("../dbg/RawLineClusters.png", resultCls);

  std::vector<uint32_t> shrinkClass;
  int nCls = (int) mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j) {
      int32_t C = resultCls.at<int32_t>(i, j);
      if (C < 0)
        throw std::logic_error("Processed label should >= 0!"
                               " in zxm::SegmentByLines()");
      resultCls.at<int32_t>(i, j) = (int) shrinkClass[C];
    }
  swap(clusterMap, resultCls);
  return nCls;
}
