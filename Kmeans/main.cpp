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

    Point<double> p0 ({1,2,3});
    Point<double> p1 ({1,5,6});
    Point<double> p2 ({1,5,3});
    Point<double> p3 ({1,5,2});
    Point<double> p4 ({1,4,5});
    Point<double> p5 ({1,5,3});
    Point<double> p6 ({1,5,2});
    Point<double> p7 ({1,4,5});
    Point<double> p8 ({100, 600, 10});
    Point<double> p9 ({90, 500, 10});
    Point<double> p10 ({100, 700, 10});
    Point<double> p11 ({90, 510, 10});
    Point<double> p12 ({100, 575, 10});
    Point<double> p13 ({90, 550, 10});
    Point<double> p14 ({1000, 2, 10});
    Point<double> p15 ({1000, 2, 10});
    Point<double> p16 ({1000, 11, 12});
    Point<double> p17 ({1000, 10, 7});
    Point<double> p18 ({1000, 2, 10});
    Point<double> p19 ({400, 200, 10});
    Point<double> p20 ({250, 250, 10});
    Point<double> p21 ({150, 175, 12});
    Point<double> p22 ({100, 50, 7});
    Point<double> p23 ({100, 2, 10});
    
    int nClusters = 3;
    // auto result = kMeansClustering<Point<double>, 
    //                                L2Distance<Point<double>>, 
    //                                PipePipeCentroidsInitializer<Point<double>, L2Distance<Point<double>>, 42>>
    //                                    (nClusters, {p1, p2, p3, p4, p5, p6, p7, p8, p10, p12, p11, p13, p15}, 0.01f); 

    auto result = PlusPlusCentroidsInitializer<Point<double>, L2Distance<Point<double>>, 42>{}(3, {p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23});
    for (const auto& [index, v] : result)
    {
        std::cout << index << std::endl; 
    }
}
