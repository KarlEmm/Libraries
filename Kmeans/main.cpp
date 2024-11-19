#include "kmeans.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

int main()
{
    using Point = KMeans::Point<double>;

    std::ifstream file("testdata.txt");
    std::string line;
    std::vector<Point> v;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        double d1;
        char comma1;
        double d2;
        char comma2;
        double d3;

        if (ss >> d1 >> comma1 >> d2 >> comma2 >> d3) 
        {
            v.push_back(Point({d1, d2, d3}));
        }
    }
    
    //KMeans::Clusters<Point> result = KMeans::kMeansClustering<Point, KMeans::L2DistanceSquared<Point>, KMeans::RandomCentroidsInitializer<Point, KMeans::L2DistanceSquared<Point>, 42>> (3, v);
    KMeans::Clusters<Point> result = KMeans::kMeansClustering<Point, KMeans::L2DistanceSquared<Point>, KMeans::PlusPlusCentroidsInitializer<Point, KMeans::L2DistanceSquared<Point>, 42>> (3, v);
    for (int i = 0; i < result.size(); ++i)
    {
        for (int j = 0; j < result[i].size(); ++j)
        {
            std::cout << "Cluster " << i << ": " << result[i][j].m_data[0] << std::endl;
        }
    }
}
