// 这里配置算法使用的超参数
// Created by zxm on 2022/12/5.
//

#ifndef ELSED_HYPERCOMFIG_H
#define ELSED_HYPERCOMFIG_H


//法向量聚类 ClusteringByNormal
constexpr float TH_CONTINUED_ANGLE = 10.f;


//结构线提取 CreateStructureLinesMap
constexpr float TH_DIFF_SIDE_ANGLE = 20.f;
constexpr float GAP_HALF_DIFF_SIDE = 1.f;
constexpr float STRUCTURE_LINE_RATIO = 0.4f;


#endif //ELSED_HYPERCOMFIG_H
