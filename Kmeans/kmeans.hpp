#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace KMeans
{

template <typename T>    
using Cluster = std::vector<T>;

template <typename T>
using Clusters = std::vector<Cluster<T>>;

template <typename T = double>
struct Point
{
    Point() = default;
    Point(const std::vector<T>& data) : m_data(data) {};
    Point(std::vector<T>&& data) : m_data(std::move(data)) {};
    Point(const Point<T>& other)
    {
        this->m_data = other.m_data;
    }
    Point(Point<T>&& other) : m_data(std::move(other.m_data)) {};
    
    Point& operator=(const Point<T>& other)
    {
        this->m_data = other.m_data;
        return *this;
    }
    Point& operator=(const std::vector<T>& data)
    {
        this->m_data = data;
        return *this;
    }

    Point& operator+=(const Point& other)
    {
        assert(m_data.size() == other.size() && "Adding Points of different dimensions.");
        auto thisItr = this->begin();
        auto otherItr = other.begin();

        for (; thisItr != this->end() && otherItr != other.end(); ++thisItr, ++otherItr)
        {
            *thisItr += *otherItr;
        }
        return *this;
    }

    bool operator==(const Point<T>& other) const
    {
        return this->m_data == other.m_data;
    }

    Point operator/(double divisor)
    {
        Point result;
        for (auto& e : m_data)
        {
            result.push_back(e / divisor);
        }
        return result;
    }

    size_t size() const { return m_data.size(); }
    std::vector<T>::iterator begin() { return m_data.begin(); }
    std::vector<T>::iterator end() { return m_data.end(); }
    
    std::vector<T>::const_iterator begin() const { return m_data.begin(); }
    std::vector<T>::const_iterator end() const { return m_data.end(); }

    void push_back(T t) { m_data.push_back(t); }

    std::vector<T> m_data;
};


template <typename T = Point<double>>
struct L2Distance
{
    double operator()(const T& left, const T& right)
    {
        assert(left.size() == right.size() && "Can't compare vectors of different sizes.");

        double squaredDistance = 0.0;
        auto leftItr = left.begin();
        auto rightItr = right.begin();

        for (; leftItr != left.end() && rightItr != right.end(); ++leftItr, ++rightItr)
        {
            squaredDistance += std::pow(*leftItr - *rightItr, 2);
        }

        return std::sqrt(squaredDistance); 
    }
};

// NOTE (keb): https://en.wikipedia.org/wiki/Earth_mover%27s_distance
template <typename T = Point<double>>
struct EMDDistance
{
    double operator()(const T& left, const T& right)
    {
        assert(left.size() == right.size() && "Can't compare vectors of different sizes.");
        size_t sz = left.size();

        double totalDistance = 0.0; 
        double previousDistance = 0.0;
        for (int i = 1; i <= sz; ++i)
        {
            previousDistance = std::abs(left[i-1] + previousDistance - right[i-1]);
            totalDistance += previousDistance;
        }

        return totalDistance;
    }
};

template <typename T, typename TDistanceFunction>
int findNearestCentroidIndex(const std::vector<T>& centroids, const T& point)
{
    TDistanceFunction distanceFunctor{};
    int nearestCentroidIndex = -1;
    int nearestCentroidDistance = std::numeric_limits<double>::max();
    for (int i = 0; i < centroids.size(); ++i)
    {
        if (auto d = distanceFunctor(centroids[i], point); d < nearestCentroidDistance)
        {
            nearestCentroidIndex = i;
            nearestCentroidDistance = d;
        }
    }

    return nearestCentroidIndex;
}

template <typename T, typename Dummy, unsigned int SEED = 0>
struct RandomCentroidsInitializer 
{
    unsigned int seed;

    RandomCentroidsInitializer()
    {
        if constexpr (SEED != 0)
        {
            seed = SEED;
        }
        else
        {
            seed = std::random_device{}();
        }
    }

    std::vector<T> operator()(int nClusters, const std::vector<T>& points)
    {
        auto pointsCpy = points;
        std::mt19937 mt{seed};
        std::shuffle(pointsCpy.begin(), pointsCpy.end(), mt);
        return {pointsCpy.begin(), pointsCpy.begin() + nClusters};
    }   
};


// NOTE: from https://en.wikipedia.org/wiki/K-means%2B%2B
template <typename T, typename TDistanceFunction, unsigned int SEED = 0, typename TContainer = std::vector<T>>
struct PlusPlusCentroidsInitializer 
{
    unsigned int seed;
    PlusPlusCentroidsInitializer()
    {
        if constexpr (SEED != 0)
        {
            seed = SEED;
        } 
        else
        {
            seed = std::random_device{}();
        }
    }

    std::vector<T> operator()(int nClusters, const TContainer& points)
    {
        std::unordered_map<int, T> centroids;
        centroids.reserve(nClusters);

        // Select a first data point at random to be the first centroid.
        std::mt19937 mt{seed};
        std::uniform_int_distribution<size_t> get_random_index(0, points.size()-1);
        size_t random_index = get_random_index(mt);
        centroids.insert({random_index, points[random_index]});

        while (centroids.size() != nClusters)
        {
            std::vector<std::pair<int, double>> probabilities = probabilitiesToBeNextCentroid(centroids, points);
            size_t random_index = selectWeightedRandomIndex(probabilities);
            centroids.insert({random_index, points[random_index]});
        }

        std::vector<T> result;
        std::transform(centroids.begin(), centroids.end(),
            std::back_inserter(result),
            [](const std::pair<int, T>& kvPair) {return kvPair.second;});
        return result;    
    }

private:
    double findNearestCentroidDistance(const T& point, const std::unordered_map<int, T>& centroids)
    {
        TDistanceFunction distanceFunctor{};
        double nearestCentroidDistance = std::numeric_limits<double>::max(); 
        for (const auto& [centroidIndex, centroid]: centroids)
        {
            if (auto d = distanceFunctor(point, centroid);
                d < nearestCentroidDistance)
            {
                nearestCentroidDistance = d; 
            }
        }
        return nearestCentroidDistance;
    }

    // For each candidate point to be the next centroid, compute the probability
    // for it to be selected based on how far it is from its closest centroid.
    // The farther it is, the more chances it has to be selected.
    std::vector<std::pair<int, double>> probabilitiesToBeNextCentroid(
        const std::unordered_map<int, T>& centroids,
        const TContainer& points)
    {
        double normalizingSum = 0.0;
        std::vector<std::pair<int, double>> weights;
        for (int i = 0; i < points.size(); ++i)
        {
            if (centroids.find(i) != centroids.end())
            {
                continue;
            }

            const auto& point = points[i];

            // NOTE (keb): This distance must really be the SQUARE of the actual distance.
            double nearestCentroidDistanceSquared = std::pow(findNearestCentroidDistance(point, centroids), 2); 
            normalizingSum += nearestCentroidDistanceSquared;
            
            assert(nearestCentroidDistanceSquared != std::numeric_limits<double>::max() && 
                "Failed to find the neareast Centroid.");
            weights.push_back({i, nearestCentroidDistanceSquared});
        }

        std::for_each(weights.begin(), weights.end(), 
            [&normalizingSum](auto& weight){weight.second = weight.second/normalizingSum;});

        return weights;
    }

    // Selects the next centroids.
    size_t selectWeightedRandomIndex(const std::vector<std::pair<int, double>>& probabilities)
    {
        std::mt19937 mt {seed};
        double rand = std::uniform_real_distribution{0.0,1.0}(mt);
        double accumulator = 0.0;
        for (int i = 0; i < probabilities.size(); ++i)
        {
            accumulator += probabilities[i].second;
            if (rand <= accumulator)
            {
                return probabilities[i].first;
            }
        }

        assert(false && "Error in probabilities computation.");
        std::unreachable();
    }
};

template <typename T>
T calculateClusterCentroid(const Cluster<T>& cluster, const T& centroid)
{
    if (cluster.empty())
    {
        return centroid;
    }

    T sum = cluster[0];
    for (size_t i = 1; i < cluster.size(); ++i)
    {
        sum += cluster[i];
    }

    return sum / cluster.size();
}

template <typename T, typename TDistanceFunction>
bool updateClustersCentroid(std::vector<T>& centroids, 
    const Clusters<T>& clusters, 
    float epsilon)
{
    TDistanceFunction distanceFunctor{};
    bool hasConverged = true;
    for (size_t i = 0; i < centroids.size(); ++i)
    {
        const T oldCentroid = centroids[i];
        centroids[i] = calculateClusterCentroid(clusters[i], oldCentroid);
        if (distanceFunctor(oldCentroid, centroids[i]) > epsilon)
        {
            hasConverged = false;
        }
    }
    return hasConverged;
}

template <typename T>
void clearClusters(Clusters<T>& clusters)
{
    for (auto& cluster : clusters)
    {
        cluster.clear();
    }
}

// Triangle Inequality Optimization - COMPARING MEANS 
// (https://www.researchgate.net/publication/225253148_Acceleration_of_K-Means_and_Related_Clustering_Algorithms)
template <typename T, typename TDistanceFunction>
std::vector<std::vector<double>> calculateInterCentroidDistances(const std::vector<T>& centroids)
{
    std::vector<std::vector<double>> distances (centroids.size(), std::vector(centroids.size(), 0.0));
    TDistanceFunction distanceFunctor {};
    for (int i = 0; i < centroids.size(); ++i)
    {
        for (int j = i+1; j < centroids.size(); ++j)
        {
            double d = distanceFunctor(centroids[i], centroids[j]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    return distances;
}

// Triangle Inequality Optimization - SORTING MEANS 
// (https://www.researchgate.net/publication/225253148_Acceleration_of_K-Means_and_Related_Clustering_Algorithms)
template <typename T, typename TDistanceFunction>
std::vector<std::vector<std::pair<int, double>>> sortInterCentroidDistances(const std::vector<std::vector<double>>& distances)
{
    std::vector<std::vector<std::pair<int, double>>> sortedDistances (distances.size(), std::vector<std::pair<int, double>> {});
    // Record the centroid index alongside its distance from the other centroid.
    for (int i = 0; i < distances.size(); ++i)
    {
        for (int j = 0; j < distances.size(); ++j)
        {
            sortedDistances[i].push_back({j, distances[i][j]});
        }
    }

    for (int i = 0; i < sortedDistances.size(); ++i)
    {
        std::sort(sortedDistances[i].begin(), sortedDistances[i].end(), 
            [](const auto& left, const auto& right){return left.second < right.second;});
    }

    return sortedDistances;
}

// NOTE: Lloyd's Algorithm from https://en.wikipedia.org/wiki/K-means_clustering
// returns the representative centroids.
template <
    typename T = Point<double>, 
    typename TDistanceFunction = L2Distance<T>, 
    typename TCentroidsInitializer = RandomCentroidsInitializer<T, TDistanceFunction>,
    typename TContainer = std::vector<T>
>
std::vector<T> kMeansClustering(int nClusters, const TContainer& points, float epsilon = 0.01)
{
    assert(nClusters <= points.size() && "Requesting more clusters than there are data points.");
    std::vector<T> centroids = TCentroidsInitializer{}(nClusters, points);
    Clusters<T> clusters(nClusters, std::vector<T>());
    std::vector<int> pointToClusterIndex (points.size(), 0);

    bool hasConverged {false};
    do
    {
        clearClusters(clusters);

        auto interCentroidDistances = calculateInterCentroidDistances<T, TDistanceFunction>(centroids);
        auto sortedInterCentroidDistances = sortInterCentroidDistances<T, TDistanceFunction>(interCentroidDistances);

        // Assign each point to the cluster with the nearest centroid.
        for (int pointIndex = 0; pointIndex < points.size(); ++pointIndex)
        {
            const auto& point = points[pointIndex];

            TDistanceFunction distanceFunctor{};

            int originalNearestClusterIndex = pointToClusterIndex[pointIndex];
            int& nearestClusterIndex = pointToClusterIndex[pointIndex];
            
            double originalNearestClusterDistance = distanceFunctor(point, centroids[originalNearestClusterIndex]);
            double nearestClusterDistance = originalNearestClusterDistance;

            // centroidIndex 0 is already processed above in nearestClusterIndex.
            for (int centroidIndex = 1; centroidIndex < centroids.size(); ++centroidIndex)
            {
                int nextCentroid = sortedInterCentroidDistances[originalNearestClusterIndex][centroidIndex].first;
                if (interCentroidDistances[originalNearestClusterIndex][nextCentroid] >= 2*originalNearestClusterDistance)
                {
                    break;
                }
                auto d = distanceFunctor(point, centroids[nextCentroid]);
                if (d < nearestClusterDistance)
                {
                    nearestClusterDistance = d;
                    nearestClusterIndex = nextCentroid;
                }
            }

            clusters[nearestClusterIndex].push_back(point);
        }

        hasConverged = updateClustersCentroid<T, TDistanceFunction>(centroids, clusters, epsilon);
    } while (!hasConverged);

    return centroids;
}

}; // namespace KMeans