#include "kmeans.hpp"

#include <gtest/gtest.h>

using namespace KMeans;

class PointTest : public testing::Test {
protected:
    PointTest() 
    {
        p1 = {1,2,3};
        p2 = {1,5,2};
        p3 = {1000, 11, 12};
        p4 = {1000, 10, 7};
        p5 = {1000, 2, 10};
        p6 = {100, 25, 10};
        p7 = {90, 20, 10};
    }

    Point<double> p1;
    Point<double> p2;
    Point<double> p3;
    Point<double> p4;
    Point<double> p5;
    Point<double> p6;
    Point<double> p7;
};

TEST_F(PointTest, opPlusEqual) 
{
    p1 += p2;
    EXPECT_EQ(Point({2,7,5}), p1);
}

TEST_F(PointTest, opDivide) 
{
    double divisor = 5.0; 
    Point pd = p4 / divisor;
    EXPECT_EQ(Point({200.0, 2.0, 1.4}), pd);
}

TEST_F(PointTest, L2Distance)
{
    L2Distance l2functor{};
    double distance = l2functor(p1, p2);
    EXPECT_NEAR(3.1622, distance, 0.0001);

    distance = l2functor(p1, p4);
    EXPECT_NEAR(999.0400, distance, 0.0001);
}

TEST_F(PointTest, calculateClusterCentroid)
{
    Cluster<Point<double>> cluster {p1, p2, p5};
    Point<double> result = calculateClusterCentroid(cluster, {});
    EXPECT_EQ(Point({334.0, 3.0, 5.0}), result);
}

TEST_F(PointTest, kMeansClusteringRandomCentroids)
{
    int nClusters = 3;
    auto result = kMeansClustering<Point<double>, 
                                   L2Distance<Point<double>>, 
                                   RandomCentroidsInitializer<Point<double>, L2Distance<Point<double>>, 42>>
                                       (nClusters, {p1, p2, p3, p4, p5, p6, p7}, 0.01f); 
    Clusters<Point<double>> expected {{p5}, {p1, p2, p6, p7}, {p3, p4}};
    EXPECT_EQ(expected, result);
}

TEST_F(PointTest, kMeansClusteringPlusPlusCentroids)
{
    int nClusters = 3;
    auto result = kMeansClustering<Point<double>, 
                                   L2Distance<Point<double>>, 
                                   PlusPlusCentroidsInitializer<Point<double>, L2Distance<Point<double>>, 42>>
                                       (nClusters, {p1, p2, p3, p4, p5, p6, p7}, 0.01f); 
    Clusters<Point<double>> expected {{p1, p2}, {p6, p7}, {p3, p4, p5}};
    EXPECT_EQ(expected, result);
}
