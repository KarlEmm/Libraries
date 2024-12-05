#include "kmeans.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

int main()
{
    using namespace KMeans;

    // std::ifstream file("testdata.txt");
    // std::string line;
    // std::vector<Point> v;
    // while (std::getline(file, line))
    // {
    //     std::stringstream ss(line);
    //     double d1;
    //     char comma1;
    //     double d2;
    //     char comma2;
    //     double d3;

    //     if (ss >> d1 >> comma1 >> d2 >> comma2 >> d3) 
    //     {
    //         v.push_back(Point({d1, d2, d3}));
    //     }
    // }
    
    //KMeans::Clusters<Point> result = KMeans::kMeansClustering<Point, KMeans::L2Distance<Point>, KMeans::RandomCentroidsInitializer<Point, KMeans::L2Distance<Point>, 42>> (6, v);
    // KMeans::Clusters<Point> result = KMeans::kMeansClustering<Point, KMeans::L2Distance<Point>, KMeans::PlusPlusCentroidsInitializer<Point, KMeans::L2Distance<Point>, 42>> (6, v);
    // for (int i = 0; i < result.size(); ++i)
    // {
    //     for (int j = 0; j < result[i].size(); ++j)
    //     {
    //         std::cout << "Cluster " << i << ": " << result[i][j].m_data[0] << std::endl;
    //     }
    // }

    Point<double> p1;
    Point<double> p2;
    Point<double> p3;
    Point<double> p4;
    Point<double> p5;
    Point<double> p6;
    Point<double> p7;
    p1 = {1,2,3};
    p2 = {1,5,2};
    p3 = {1000, 11, 12};
    p4 = {1000, 10, 7};
    p5 = {1000, 2, 10};
    p6 = {100, 25, 10};
    p7 = {90, 20, 10};
    
    int nClusters = 3;
    auto result = kMeansClustering<Point<double>, 
                                   L2Distance<Point<double>>, 
                                   PipePipeCentroidsInitializer<Point<double>, L2Distance<Point<double>>, 42>>
                                       (nClusters, {p1, p2, p3, p4, p5, p6, p7}, 0.01f); 

    // std::vector<Point<double>> expected {Point({1, 3.5, 2.5}), Point({95, 22.5, 10}), Point({1000, 23/3.0, 29/3.0})};
}
