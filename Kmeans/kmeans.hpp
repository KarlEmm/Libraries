#pragma once

#include <random.hpp>
#include <thread.hpp>
#include <time.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream> // TODO (keb): remove, only for debugging
#include <mutex>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace KMeans
{

using PointIndex = uint32_t;

constexpr int nThreads = 12;

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
    
    Point& operator*=(double factor)
    {
        for (auto& e : m_data)
        {
            e *= factor;
        }
        return *this;
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

    const T& operator[] (size_t index) const { return m_data[index]; }
    T& operator[](size_t index) { return m_data[index]; }
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

        double result = 0.0;
        std::vector<double> emd (sz + 1, 0);
        for (int i = 1; i <= sz; ++i)
        {
            emd[i] = left[i-1] + emd[i-1] - right[i-1];
            result += std::abs(emd[i]);
        }

        return result;
    }
};

template <typename T, typename TDistanceFunction, typename TContainer>
std::pair<int, double> findNearestCentroid(const TContainer& centroids, const T& point)
{
    TDistanceFunction distanceFunctor{};
    int nearestCentroidIndex = -1;
    double nearestCentroidDistance = std::numeric_limits<double>::max();
    for (int i = 0; i < centroids.size(); ++i)
    {
        if (auto d = distanceFunctor(centroids[i], point); d < nearestCentroidDistance)
        {
            nearestCentroidIndex = i;
            nearestCentroidDistance = d;
        }
    }
    
    assert(nearestCentroidIndex != -1 && "Couldn't find a nearest centroid.");

    return {nearestCentroidIndex, nearestCentroidDistance};
}

template <typename T, typename TDistanceFunction, typename TContainer>
double clusterCost(const TContainer& points, const std::vector<T>& centroids)
{
    double result = 0.0;
    for (const auto& point : points)
    {
        auto [nearestCentroidIndex, nearestCentroidDistance] = findNearestCentroid<T, TDistanceFunction>(centroids, point);
        result += std::pow(nearestCentroidDistance, 2); 
    }
    return result;
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

template <
    typename T = Point<double>, 
    typename TDistanceFunction = L2Distance<T>, 
    typename TContainer = std::vector<T>
>
std::pair<bool, double> assignPointsToCluster(Clusters<PointIndex>& clusters, 
    std::vector<PointIndex>& pointToClusterIndex,
    const std::vector<T>& centroids,
    const TContainer& points)
{
    clearClusters(clusters);

    auto interCentroidDistances = calculateInterCentroidDistances<T, TDistanceFunction>(centroids);
    auto sortedInterCentroidDistances = sortInterCentroidDistances<T, TDistanceFunction>(interCentroidDistances);

    std::atomic<bool> hasOnePointChangedCluster {false};
    std::atomic<double> error {0.0};
    std::mutex mutex;

    // Assign each point to the cluster with the nearest centroid.
    auto callback = [&clusters, &points, &pointToClusterIndex, &interCentroidDistances, &sortedInterCentroidDistances, &centroids, &hasOnePointChangedCluster, &error, &mutex](size_t start, size_t end, int threadId)
    {
        Clusters<PointIndex> threadClusters (clusters.size(), std::vector<PointIndex>());
        auto startTime = std::chrono::high_resolution_clock::now();
        for (PointIndex pointIndex = start; pointIndex < end; ++pointIndex)
        {
            if ((pointIndex-start) % 1'000'000 == 0 && threadId == 0)
            {
                std::cout << "Iteration Time Remaining: " << Time::timeRemaining(end-start, pointIndex-start, startTime) << "s" << std::endl;
            }
            const auto& point = points[pointIndex];

            TDistanceFunction distanceFunctor{};

            PointIndex& nearestClusterIndex = pointToClusterIndex[pointIndex];
            const PointIndex originalNearestClusterIndex = nearestClusterIndex;
            
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
                    hasOnePointChangedCluster = true;
                    nearestClusterDistance = d;
                    nearestClusterIndex = nextCentroid;
                }
            }

            threadClusters[nearestClusterIndex].push_back(pointIndex);

            error += std::pow(nearestClusterDistance, 2);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (int clusterIndex = 0; clusterIndex < threadClusters.size(); ++clusterIndex)
            {
                clusters[clusterIndex] = std::move(threadClusters[clusterIndex]);
            }
        }
    };

    Thread::startThreadedLoop(callback, points.size(), nThreads);

    return {!hasOnePointChangedCluster, error};
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

    std::unordered_map<size_t, T> operator()(int nClusters, const std::vector<T>& points)
    {
        auto pointsCpy = points;
        std::mt19937 mt{seed};
        std::shuffle(pointsCpy.begin(), pointsCpy.end(), mt);
        std::unordered_map<size_t, T> result;
        for (int i = 0; i < nClusters; ++i)
        {
            // NOTE (keb): the centroid index is irrelevant with the Random Initializer.
            result.insert({i, pointsCpy[i]});
        }
        return result;
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

    std::unordered_map<size_t, T> operator()(int nClusters, const TContainer& points)
    {
        std::unordered_map<size_t, T> centroids;
        centroids.reserve(nClusters);

        // Select a first data point at random to be the first centroid.
        std::mt19937 mt{seed};
        std::uniform_int_distribution<size_t> get_random_index(0, points.size()-1);
        size_t random_index = get_random_index(mt);
        centroids.insert({random_index, points[random_index]});

        std::cout << "PlusPlus Selecting Centroids" << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        while (centroids.size() != nClusters)
        {
            std::vector<std::pair<PointIndex, double>> probabilities = probabilitiesToBeNextCentroid(centroids, points);
            PointIndex random_index = selectWeightedRandomIndex(probabilities);
            centroids.insert({random_index, points[random_index]});
            std::cout << centroids.size() << "/" << nClusters << " centroids found." << std::endl;
            std::cout << "Time Remaining: " << Time::timeRemaining(nClusters, centroids.size(), startTime) << "s" << std::endl;
        }

        return centroids;
    }

private:
    double findNearestCentroidDistance(const T& point, const std::unordered_map<size_t, T>& centroids)
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
    std::vector<std::pair<PointIndex, double>> probabilitiesToBeNextCentroid(
        const std::unordered_map<size_t, T>& centroids,
        const TContainer& points,
        const int lambda = 1)
    {
        double normalizingSum = 0.0;
        std::vector<std::pair<PointIndex, double>> weights;
        weights.reserve(points.size());
        for (PointIndex i = 0; i < points.size(); ++i)
        {
            if (centroids.find(i) != centroids.end())
            {
                continue;
            }

            const auto& point = points[i];

            // NOTE (keb): This distance must really be the SQUARE of the actual distance.
            double nearestCentroidDistanceSquared = std::pow(findNearestCentroidDistance(point, centroids), 2); 
            normalizingSum += nearestCentroidDistanceSquared;
            nearestCentroidDistanceSquared *= lambda; 
            
            assert(nearestCentroidDistanceSquared != std::numeric_limits<double>::max() && 
                "Failed to find the neareast Centroid.");
            weights.push_back({i, nearestCentroidDistanceSquared});
        }

        if (normalizingSum > 0.0)
        {
            std::for_each(weights.begin(), weights.end(), 
                [&normalizingSum](auto& weight){weight.second = weight.second/normalizingSum;});
        }
        else
        {
            // NOTE (keb): no option is good, pick any point as the next centroid.
            std::for_each(weights.begin(), weights.end(), 
                [&weights](auto& weight){weight.second = 1.0 / weights.size();});
        }

        return weights;
    }

    // Selects the next centroids.
    PointIndex selectWeightedRandomIndex(const std::vector<std::pair<PointIndex, double>>& probabilities)
    {
        std::mt19937 mt {seed};
        double rand = std::uniform_real_distribution{0.0,1.0}(mt);
        double accumulator = 0.0;
        for (const auto& [pointIndex, probability] : probabilities)
        {
            accumulator += probability;
            if (rand <= accumulator)
            {
                return pointIndex;
            }
        }

        assert(false && "Error in probabilities computation.");
        std::unreachable();
    }
};


template <typename T, typename TContainer>
T calculateClusterCentroid(const Cluster<PointIndex>& cluster, const T& centroid, const TContainer& points)
{
    if (cluster.empty())
    {
        return centroid;
    }

    T sum = points[cluster[0]];
    for (size_t i = 1; i < cluster.size(); ++i)
    {
        sum += points[cluster[i]];
    }

    return sum / cluster.size();
}

template <typename T, typename TDistanceFunction, typename TContainer>
void updateClustersCentroid(std::vector<T>& centroids, 
    const Clusters<PointIndex>& clusters,
    const TContainer& points)
{
    auto callback = [&clusters, &centroids, &points](size_t start, size_t end, int threadId)
    {
        for (size_t i = start; i < end; ++i)
        {
            const T oldCentroid = centroids[i];
            centroids[i] = calculateClusterCentroid<T, TContainer>(clusters[i], oldCentroid, points);
        }
    };

    Thread::startThreadedLoop(callback, centroids.size(), nThreads);
}

// TODO (keb): Convergence is problematic. Potentially add condition: if no point changes cluster -> converged
// NOTE: Lloyd's Algorithm from https://en.wikipedia.org/wiki/K-means_clustering
// returns the representative centroids.
template <
    typename T = Point<double>, 
    typename TDistanceFunction = L2Distance<T>, 
    typename TCentroidsInitializer = RandomCentroidsInitializer<T, TDistanceFunction>,
    typename TContainer = std::vector<T>
>
std::vector<T> kMeansClustering(int nClusters, const TContainer& points, int nIterations = 35)
{
    assert(nClusters <= points.size() && "Requesting more clusters than there are data points.");
    int nRuns = 2;
    std::vector<std::vector<T>> centroidsValuesCandidates;
    std::pair<size_t, double> bestCandidate = {0, std::numeric_limits<double>::max()};

    for (int runIndex = 0; runIndex < nRuns; ++runIndex)
    {
        auto centroids = TCentroidsInitializer{}(nClusters, points);
        Clusters<PointIndex> clusters(nClusters, std::vector<PointIndex>());
        std::vector<PointIndex> pointToClusterIndex (points.size(), 0);

        std::vector<T> centroidsValues;
        centroidsValues.reserve(centroids.size());
        std::transform(centroids.begin(), centroids.end(),
            std::back_inserter(centroidsValues),
            [](const auto& kv){ return kv.second; });

        double totalError = std::numeric_limits<double>::max();
        int iteration = 0;
        std::cout << std::endl;
        std::cout << "Lloyd's Run: " << runIndex+1 << "/" << nRuns << std::endl;
        do
        {
            auto [hasConverged, error] = assignPointsToCluster<T, TDistanceFunction, TContainer>(clusters, pointToClusterIndex, centroidsValues, points);
            if (hasConverged)
            {
                totalError = error;
                break;
            }
            updateClustersCentroid<T, TDistanceFunction, TContainer>(centroidsValues, clusters, points);
            ++iteration;
        } while (iteration < nIterations);

        centroidsValuesCandidates.push_back(std::move(centroidsValues));
        if (totalError <= bestCandidate.second)
        {
            bestCandidate = {runIndex, totalError};
        }
    }

    return centroidsValuesCandidates[bestCandidate.first];
}

// NOTE: from https://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf
template <typename T, typename TDistanceFunction, unsigned int SEED = 0, typename TContainer = std::vector<T>>
struct PipePipeCentroidsInitializer 
{
    unsigned int seed;
    PipePipeCentroidsInitializer()
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

    std::unordered_map<size_t, T> operator()(int nClusters, const TContainer& points)
    {
        std::unordered_map<size_t, T> centroids;
        centroids.reserve(nClusters);

        // Select a first data point at random to be the first centroid.
        std::mt19937 mt{seed};
        std::uniform_int_distribution<size_t> get_random_index(0, points.size()-1);
        size_t random_index = get_random_index(mt);
        centroids.insert({random_index, points[random_index]});
        
        // NOTE (keb): log(phi) is a theoretical bound.
        // double phi = clusterCost<T, TDistanceFunction, TContainer>(points, {centroids[random_index]});
        // int nIterations = std::log(phi) + 1;
        // NOTE (keb): 8 has been found to be good experimentally.
        // Here, I use r = 5 and l = 2*k.
        constexpr int nIterations = 5;
        int lambda = nClusters * 2;

        std::cout << std::endl;
        std::cout << "PipePipe Selecting Intermediate Centroids" << std::endl;
        for (int i = 0; i < nIterations; ++i)
        {
            std::cout << "Iteration: " << i+1 << "/" << nIterations << std::endl;
            std::vector<double> probabilities = probabilitiesToBeNextCentroid(centroids, points, lambda);
            selectWeightedRandomIndex(probabilities, nClusters, points, centroids); 
            std::cout << "Selected " << centroids.size() << " intermediate centroids in total." << std::endl << std::endl;
        }

        Clusters<PointIndex> clusters(centroids.size(), std::vector<PointIndex>());
        std::vector<PointIndex> pointToClusterIndex (points.size(), 0); 
        std::vector<size_t> centroidsKeys;
        std::vector<T> centroidsValues;
        centroidsKeys.reserve(centroids.size());
        centroidsValues.reserve(centroids.size());
        for (const auto& [pointIndex, centroid] : centroids)
        {
            centroidsKeys.push_back(pointIndex);
            centroidsValues.push_back(centroid);
        }

        std::cout << "Computing Intermediate Centroids Weight" << std::endl;
        assignPointsToCluster<T, TDistanceFunction, TContainer>(clusters, pointToClusterIndex, centroidsValues, points);

        for (int i = 0; i < centroidsValues.size(); ++i)
        {
            // NOTE (keb): a cluster can be empty if two centroids are identical.
            centroidsValues[i] *= (clusters[i].empty() ? 1.0 : clusters[i].size());
        }

        auto chosenCentroids = PlusPlusCentroidsInitializer<T, TDistanceFunction, SEED>{}(nClusters, centroidsValues);
        std::unordered_map<size_t, T> result;
        for (const auto& [index, centroid] : chosenCentroids)
        {
            result.insert({centroidsKeys[index], centroidsValues[index] / (clusters[index].empty() ? 1.0 : clusters[index].size())});
        }
        return result;
    }

private:
    double findNearestCentroidDistance(const T& point, const std::unordered_map<size_t, T>& centroids)
    {
        TDistanceFunction distanceFunctor{};
        double nearestCentroidDistance = std::numeric_limits<double>::max(); 
        for (const auto& [pointIndex, centroid]: centroids)
        {
            if (auto d = distanceFunctor(point, centroid);
                d < nearestCentroidDistance)
            {
                nearestCentroidDistance = d; 
            }
        }
        return nearestCentroidDistance;
    }

    std::vector<double> probabilitiesToBeNextCentroid(
        const std::unordered_map<size_t, T> centroids,
        const TContainer& points,
        const int lambda = 1)
    {
        std::atomic<double> normalizingSum = 0.0;
        std::vector<double> weights (points.size(), 0);

        auto callback = [this, &points, &centroids, &weights, &normalizingSum, &lambda](size_t start, size_t end, int threadId)
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            for (int i = start; i < end; ++i)
            {
                if ((i-start) % 1'000'000 == 0 && threadId == 0)
                {
                    std::cout << "Iteration Time Remaining: " << Time::timeRemaining(end-start, i-start, startTime) << "s" << std::endl;
                }
                
                const auto& point = points[i];

                // NOTE (keb): This distance must really be the SQUARE of the actual distance.
                double nearestCentroidDistanceSquared = std::pow(findNearestCentroidDistance(point, centroids), 2); 
                normalizingSum += nearestCentroidDistanceSquared;

                // NOTE (keb): Scale by lambda to select lambda centroids in expectation (see article).
                nearestCentroidDistanceSquared *= lambda;
                
                assert(nearestCentroidDistanceSquared != std::numeric_limits<double>::max() && 
                    "Failed to find the neareast Centroid.");
                weights[i] = nearestCentroidDistanceSquared;
            }
        };

        Thread::startThreadedLoop(callback, points.size(), nThreads); 

        std::for_each(weights.begin(), weights.end(), 
            [&normalizingSum](auto& weight){weight = weight/normalizingSum;});

        return weights;
    }

    // Selects the next centroids.
    void selectWeightedRandomIndex(const std::vector<double>& probabilities,
        int nClusters,
        const TContainer& points,
        std::unordered_map<size_t, T>& centroids)
    {
        std::mutex mutex;
        auto callback = [&probabilities, &nClusters, &points, &centroids, &mutex](size_t start, size_t end, int threadid)
        {
            std::unordered_map<size_t, T> threadCentroids;
            threadCentroids.reserve(nClusters);

            std::mt19937 mt{ Random::generate() };
            std::uniform_real_distribution dist { 0.0, 1.0 };
            auto startTime = Time::now();
            for (int pointIndex = start; pointIndex < end; ++pointIndex)
            {
                if ((pointIndex-start) % 1'000'000 == 0 && threadid == 0)
                {
                    std::cout << "Sampling Intermediate Centroids Time Remaining: " << Time::timeRemaining(end-start, pointIndex-start, startTime) << "s" << std::endl;
                }
                double p = dist(mt);
                if (p <= probabilities[pointIndex])
                {
                    threadCentroids.insert({pointIndex, points[pointIndex]});
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (auto& kv : threadCentroids)
                {
                    centroids.insert(std::move(kv));
                }
            }
        };
        
        Thread::startThreadedLoop(callback, points.size(), nThreads);
        for (int i = 0; i < 200; ++i)
        {
            centroids.insert({i, points[i]});
        }
    }
    
    void reweightProbabilities(std::unordered_map<int, double>& probabilities, PointIndex indexToRemove)
    {
        double weightToRemove = probabilities[indexToRemove];
        double normalizingSum = 1.0 - weightToRemove;
        probabilities[indexToRemove] = 0.0;
        for (auto& [pointIndex, probability]: probabilities)
        {
            probability /= normalizingSum;
        }
    }
};

}; // namespace KMeans