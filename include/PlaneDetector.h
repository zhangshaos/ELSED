//
// Created by zxm on 2022/12/5.
//

#ifndef ELSED_PLANEDETECTOR_H
#define ELSED_PLANEDETECTOR_H


#include <opencv2/core.hpp>


namespace zxm
{

/**
 * 检测图片中的所有平面区域，并返回这些区域的遮罩（一般分辨率为256x192）。
 * @param[in] colorImg CV_8UC3 一般为960x720的RGB图片
 * @param[in] normalMap CV_32FC3 一般为256x192的法向量图
 * @param[in] enableDebug 开启后，会在可执行文件的上层文件夹中创建dbg目录(即"../dbg/")，并记录算法中间数据：
 * @note 法向量图为归一化向量，且不含零向量
 *
 * 1. NormalClusters.png 法向量聚类中间结果\n
 * 2. Lines.png 结构线检测结果\n
 * 3. BlendNormalClustersAndLines.png 混合结构线和法向量聚类结果
 * @return CV_32S 记录每个像素所属的平面ID
 */
  cv::Mat DetectPlanes(const cv::Mat &colorImg,
                       const cv::Mat &normalMap,
                       bool enableDebug=false);


/**
 * 将normalMap按照法向量和像素联通关系（8邻域）进行聚类，并将结果写入clusterMap中
 * @param[out] clusterMap CV_32S
 * @param[in] normalMap CV_32FC3 归一化的法向量图（不含零向量）
 * @return 类别数量
 */
  int ClusteringByNormal(cv::Mat &clusterMap, const cv::Mat &normalMap);


/**
 * 运行ELSED算法检测RGB图片colorMap中的所有线段。
 * @note 1. 此函数会筛除掉那些纹理线：假设纹理线上所有像素两侧法向量一致完全一致。
 * @note 2. colorImg的尺寸可以大于normalMap。
 * @param[in] colorImg CV_8UC3
 * @param[in] normalMap CV_32FC3 归一化的法向量图（不含零向量）
 * @return cv:Mat CV_32S 结构线遮罩：>0表示线段，不同的数字表示不同的线段ID
 */
  cv::Mat CreateStructureLinesMap(const cv::Mat &colorImg,
                                  const cv::Mat &normalMap);


/**
 * 给定一个初步聚类图clusterMap，使用检测到的结构线lineMap，将聚类图的簇进一步分割开来。
 * @param[in&out] clusterMap CV_32S
 * @param[in] lineMap CV_32S
 * @return 最终分割图的类别数量
 */
  int SegmentByLines(cv::Mat &clusterMap,
                     const cv::Mat &lineMap,
                     bool enableDebug=false);


}



#endif //ELSED_PLANEDETECTOR_H
