#include <algorithm>
#include <cassert>
#include <iostream> // TODO (keb): for temporary dedugging.
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
    Point(const std::vector<T>& data) : m_data(data) 
    {

    };
    Point(std::vector<T>&& data) : m_data(std::move(data)) 
    {

    };
    Point(const Point<T>& other)
    {
        this->m_data = other.m_data;
    }
    Point(Point<T>&& other) : m_data(std::move(other.m_data)) 
    {

    };
    
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
struct L2DistanceSquared
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

        return squaredDistance; 
    }
};

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
template <typename T, typename TDistanceFunction, unsigned int SEED = 0>
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

    std::vector<T> operator()(int nClusters, const std::vector<T>& points)
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
        const std::vector<T>& points)
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
            double nearestCentroidDistance = findNearestCentroidDistance(point, centroids); 
            normalizingSum += nearestCentroidDistance;
            
            assert(nearestCentroidDistance != std::numeric_limits<double>::max() && 
                "Failed to find the neareast Centroid.");
            weights.push_back({i, nearestCentroidDistance});
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

// NOTE: Lloyd's Algorithm from https://en.wikipedia.org/wiki/K-means_clustering
template <
    typename T = Point<double>, 
    typename TDistanceFunction = L2DistanceSquared<T>, 
    typename TCentroidsInitializer = RandomCentroidsInitializer<T, TDistanceFunction>
>
Clusters<T> kMeansClustering(int nClusters, const std::vector<T>& points, float epsilon = 0.01)
{
    assert(nClusters <= points.size() && "Requesting more clusters than there are data points.");
    std::vector<T> centroids = TCentroidsInitializer{}(nClusters, points);
    Clusters<T> clusters(nClusters, std::vector<T>());

    bool hasConverged {false};
    do
    {
        clearClusters(clusters);

        // Assign each point to the cluster with the nearest centroid.
        for (const auto& point : points)
        {
            std::vector<double> pointToCentroidDistances;
            TDistanceFunction distanceFunctor{};
            std::transform(centroids.begin(), centroids.end(), std::back_inserter(pointToCentroidDistances), 
                [&point, &distanceFunctor](const auto& centroid){ return distanceFunctor(point, centroid); });

            auto pointClosestCentroidItr = std::min_element(pointToCentroidDistances.begin(), 
                pointToCentroidDistances.end());

            std::ptrdiff_t closestClusterIndex = std::distance(pointToCentroidDistances.begin(), pointClosestCentroidItr);
            clusters[closestClusterIndex].push_back(point);
        }

        hasConverged = updateClustersCentroid<T, TDistanceFunction>(centroids, clusters, epsilon);
    } while (!hasConverged);

    return clusters;
}

}; // namespace KMeans