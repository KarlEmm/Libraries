#pragma once

#include <pokerTypes.hpp>
#include <kmeans.hpp>
#include <memory.hpp>

#include <phevaluator/phevaluator.h>
extern "C"
{
    #include <hand_index.h>
}

#include <x86intrin.h>
#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream> // TODO: remove after debugging
#include <memory>
#include <sys/mman.h>
#include <thread>
#include <unordered_set>
#include <vector>


// NOTE: Library to compute Texas Hold'em Effective Hand Strength
// as done in https://en.wikipedia.org/wiki/Effective_hand_strength_algorithm
// EHS = HS * (1 - NPOT) + (1 - HS) * PPOT

// NOTE: Compute Hand Rank for 5, 6, and 7 card hands with methods found in 
// https://github.com/HenryRLee/PokerHandEvaluator

namespace EHS
{
    using namespace PokerTypes;

    constexpr int nThreads = 12;


    struct HandStats
    {
        std::atomic<uint32_t> winner {0};
        std::atomic<uint32_t> loser {0};
        std::atomic<uint32_t> draw {0};
    };

    constexpr uint64_t factorial(int n, int qty)
    {
        uint64_t result = 1;
        for (int i = 0; i < qty; ++i)
        {
            result *= (n-i);
        }
        return result;
    }

    inline float getEHS(const std::array<PokerTypes::Card, 2>& myCards, const std::array<PokerTypes::Card, 5>& boardCards, const std::bitset<52>& deck)
    {
        using namespace PokerTypes;
        HandStats stats;
        int myRank = evaluate_7cards(myCards[0], myCards[1], boardCards[0], boardCards[1], boardCards[2], boardCards[3], boardCards[4]);

        for (int o1 = 0; o1 < 52; ++o1)
        {
            if (deck.test(o1)) [[unlikely]]
                continue;
            for (int o2 = o1+1; o2 < 52; ++o2)
            {
                if (deck.test(o2)) [[unlikely]]
                    continue;

                int theirRank = evaluate_7cards(o1, o2, boardCards[0], boardCards[1], boardCards[2], boardCards[3], boardCards[4]);

                myRank < theirRank ? ++stats.winner : myRank > theirRank ? ++stats.loser : ++stats.draw;
            }
        }

        float result = (stats.winner + (stats.draw / 2.0)) / (stats.winner + stats.loser + stats.draw);
        return result;
    }
    
    inline Histogram<AbstractionsContext::nOCHSHistogramsBins> getOCHS(const std::array<PokerTypes::Card, 2>& myCards, 
        const std::array<PokerTypes::Card, 5>& boardCards, 
        const std::bitset<52>& deck)
    {
        using namespace PokerTypes;
        const int myRank = evaluate_7cards(myCards[0], myCards[1], boardCards[0], boardCards[1], boardCards[2], boardCards[3], boardCards[4]);

        HandStats stats[AbstractionsContext::nOCHSHistogramsBins];
        Histogram<AbstractionsContext::nOCHSHistogramsBins> result;

        for (int ochsIndex = 0; ochsIndex < AbstractionsContext::nOCHSHistogramsBins; ++ochsIndex)
        {
            for (const auto& cards : AbstractionsContext::preflopOCHSBuckets[ochsIndex])
            {
                if (deck.test(cards.first) || deck.test(cards.second)) [[unlikely]]
                {
                    continue;
                }
                const int theirRank = evaluate_7cards(cards.first, cards.second, boardCards[0], boardCards[1], boardCards[2], boardCards[3], boardCards[4]);

                myRank < theirRank ? ++stats[ochsIndex].winner : myRank > theirRank ? ++stats[ochsIndex].loser : ++stats[ochsIndex].draw;
            }
        }

        for (int ochsIndex = 0; ochsIndex < AbstractionsContext::nOCHSHistogramsBins; ++ochsIndex)
        {
            result[ochsIndex] = (stats[ochsIndex].winner + (stats[ochsIndex].draw / 2.0)) / (stats[ochsIndex].winner + stats[ochsIndex].loser + stats[ochsIndex].draw);
        }
        return result;
    }

    constexpr int nCr(int n, int r)
    {
        //    n!
        // --------
        // (n-r)!r!
        int numerator = 1;
        for (int i = n; i > (n-r); --i)
        {
            numerator *= i;
        }

        return numerator / factorial(r, r);
    }

    static_assert(nCr(7,2) == 21);
    static_assert(nCr(6,2) == 15);
    static_assert(nCr(5,2) == 10);

    template <typename E>
    int enumToInt(E e)
    {
        return static_cast<int>(e);
    }

    void generateFlopHistograms()
    {
        hand_indexer_t turn_indexer;
        assert(hand_indexer_init(3, (const uint8_t[]){2,3,1}, &turn_indexer));
        uint64_t nCanonicalHandsTurn = hand_indexer_size(&turn_indexer, 2);
        auto turnHistograms = Memory::getMmap<Histogram<AbstractionsContext::nEHSHistogramsBins>>(AbstractionsContext::turnHistogramsFilename, nCanonicalHandsTurn);
        if (!turnHistograms) return;
        
        uint64_t nCanonicalHandsFlop = hand_indexer_size(&turn_indexer, 1);
        auto flopHistograms = Memory::getMmap<Histogram<AbstractionsContext::nEHSHistogramsBins>>(AbstractionsContext::flopHistogramsFilename, nCanonicalHandsFlop, false);
        if (!flopHistograms) return;

        std::cout << std::endl;
        std::cout << "Generating Flop EHS Histograms" << std::endl;
        for (int canonicalIndex = 0; 
            canonicalIndex < nCanonicalHandsFlop; 
            ++canonicalIndex)
        {
            uint8_t cards[5];
            if (!hand_unindex(&turn_indexer, 1, canonicalIndex, cards))
            {
                return;
            }

            std::bitset<52> deck;
            for (int i = 0; i < 5; ++i)
            {
                deck.flip(cards[i]);
            }

            Histogram histogram; 
            for (int i = 0; i < 52; ++i)
            {
                if (deck.test(i))
                {
                    continue;
                }

                uint8_t turnCards[6];
                std::memcpy(turnCards, cards, 5);
                turnCards[5] = i;
                uint64_t turnIndex = hand_index_last(&turn_indexer, turnCards);
                
                histogram += turnHistograms[turnIndex];
            }
            flopHistograms[canonicalIndex] = std::move(histogram);
        }
    }

    // NOTE (keb): a histogram keeps track of the number of times a hand ends up
    // with a given strength on the river. The bins intervals represent the strength.
    // The bins values represent the number of time at that strength.
    void generateTurnHistograms()
    {
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
        
        uint64_t nCanonicalHandsRiver = hand_indexer_size(&river_indexer, 3);
        uint64_t nEHS = nCanonicalHandsRiver;
        auto riverEHS = Memory::getMmap<uint16_t>(AbstractionsContext::riverEHSFilename, nEHS);
        if (!riverEHS) return;

        uint64_t nCanonicalHandsTurn = hand_indexer_size(&river_indexer, 2);
        uint64_t nHistograms = nCanonicalHandsTurn;
        auto turnHistograms = Memory::getMmap<Histogram<AbstractionsContext::nEHSHistogramsBins>>(AbstractionsContext::turnHistogramsFilename, nHistograms, false);
        if (!turnHistograms) return;

        std::cout << std::endl;
        std::cout << "Generating Turn EHS Histograms" << std::endl;
        auto callback = [&river_indexer, &turnHistograms, &riverEHS](uint32_t start, uint32_t end, int threadid)
        {
            uint64_t total = end-start;
            auto startChrono = std::chrono::high_resolution_clock::now();

            for (int canonicalIndex = start; 
                canonicalIndex < end; 
                ++canonicalIndex)
            {
                if (uint32_t done = canonicalIndex-start; done % 10'000'000 == 0 && threadid == 0)
                {
                    std::cout << "Time Remaining " << Time::timeRemaining(total, done, startChrono) << "(s)" << std::endl;
                }
                
                uint8_t cards[6];
                if (!hand_unindex(&river_indexer, 2, canonicalIndex, cards))
                {
                    return;
                }

                std::bitset<52> deck;
                for (int i = 0; i < 6; ++i)
                {
                    deck.flip(cards[i]);
                }
                
                Histogram histogram;
                for (int i = 0; i < 52; ++i)
                {
                    if (deck.test(i))
                    {
                        continue;
                    }
                    uint8_t riverCards[7];
                    std::memcpy(riverCards, cards, 6);
                    // NOTE (keb): found a 7th card to complete a river hand.
                    riverCards[6] = i;
                    uint64_t riverIndex = hand_index_last(&river_indexer, riverCards);
                    
                    uint16_t riverStrength = riverEHS[riverIndex];
                    double rhs = riverStrength / 10'000.0f;
                    // NOTE (keb): histogram has x bins
                    // 0.05 0.10 0.15 0.20 0.25 0.30 0.35 ... 0.95 1.00
                    // [0,0.05), [0.05, 0.10), ..., [0.95, 1.00]
                    size_t histogramSize = histogram.size();
                    int binIndex = rhs*histogramSize;
                    histogram.incrementBin(binIndex == histogramSize ? histogramSize-1 : binIndex, 1);
                }
                turnHistograms[canonicalIndex] = std::move(histogram);
            }
        };
        
        Thread::startThreadedLoop(callback, nCanonicalHandsTurn, nThreads);
    }

    void finalizeHistograms(int round, const char* filename)
    {
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));

        uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, round);
        auto histograms = Memory::modifyMmap<Histogram<AbstractionsContext::nEHSHistogramsBins>>(filename, nCanonicalHands);

        for (size_t i = 0; i < nCanonicalHands; ++i)
        {
            if (!histograms[i].isConvertedToPercent)
            {
                histograms[i].convertToPercent();
            }
        }
    }

    void generateRoundsHistograms()
    {
        generateTurnHistograms();
        generateFlopHistograms();
        
        finalizeHistograms(1, AbstractionsContext::flopHistogramsFilename);
        finalizeHistograms(2, AbstractionsContext::turnHistogramsFilename);
    }

    template <int NHistogramBins, typename TDistanceFunction>
    void writeRoundCentroids(const char* centroidsFilename, 
        const char* histogramFilename,
        int nAbstractionBuckets, 
        int roundIndex)
    {
        std::cout << std::endl;
        std::cout << "Computing and writing centroids for " << centroidsFilename << std::endl;
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
        uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, roundIndex);
        auto histograms = Memory::getMmap<Histogram<NHistogramBins>>(histogramFilename, nCanonicalHands);

        std::vector<Histogram<NHistogramBins>> result (nAbstractionBuckets);
        std::span<Histogram<NHistogramBins>> dataPoints (histograms.get(), histograms.get() + nCanonicalHands);
        result = KMeans::kMeansClustering<Histogram<NHistogramBins>, 
            TDistanceFunction, 
            KMeans::PipePipeCentroidsInitializer<Histogram<NHistogramBins>, TDistanceFunction, 0, std::span<Histogram<NHistogramBins>>>
            > (nAbstractionBuckets, dataPoints);

        
        auto centroids = Memory::getMmap<Histogram<NHistogramBins>>(centroidsFilename, nAbstractionBuckets, false);
        for (int i = 0; i < nAbstractionBuckets; ++i)
        {
            centroids[i] = std::move(result[i]);
        }
    }

    void writeRoundsCentroids()
    {
        writeRoundCentroids<AbstractionsContext::nEHSHistogramsBins, KMeans::EMDDistance<Histogram<AbstractionsContext::nEHSHistogramsBins>>>(AbstractionsContext::flopCentroidsFilename,
            AbstractionsContext::flopHistogramsFilename, 
            AbstractionsContext::nBuckets, 1); 
        
        writeRoundCentroids<AbstractionsContext::nEHSHistogramsBins, KMeans::EMDDistance<Histogram<AbstractionsContext::nEHSHistogramsBins>>>(AbstractionsContext::turnCentroidsFilename,
            AbstractionsContext::turnHistogramsFilename, 
            AbstractionsContext::nBuckets, 2);
        
        writeRoundCentroids<AbstractionsContext::nOCHSHistogramsBins, KMeans::L2Distance<Histogram<AbstractionsContext::nOCHSHistogramsBins>>>(AbstractionsContext::riverCentroidsFilename,
            AbstractionsContext::riverOCHSFilename, 
            AbstractionsContext::nBuckets, 3);
    }

    void generateRiverEHS()
    {
        using namespace PokerTypes;
        
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));

        uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, 3);
        uint64_t nEHS = nCanonicalHands;
        // NOTE (keb): a EHS is number between 0 and 1 representing the probability
        // a hand will win against a random hand pick according to a uniform distribution.
        // We use uint16_t to store a EHS. To map the stored value to a number between 0 and 1,
        // divide the stored value by 10'000.
        auto riverEHS = Memory::getMmap<uint16_t>(AbstractionsContext::riverEHSFilename, nEHS, false);
        if (!riverEHS) return;

        std::cout << std::endl;
        std::cout << "Generating River EHS" << std::endl;
        auto callback = [&river_indexer, &riverEHS](uint32_t start, uint32_t end, int threadid)
        {
            auto startChrono = std::chrono::high_resolution_clock::now();
            std::array<Card, 2> privateCards;
            std::array<Card, 5> publicCards;

            for (uint32_t i = start; i < end; ++i)
            {
                if (uint32_t done = i-start; done % 10'000'000 == 0 && threadid == 0)
                {
                    std::cout << "Time Remaining: " << Time::timeRemaining(end-start, i-start, startChrono) << " (s)" << std::endl;
                }

                uint8_t cards[7];
                if (!hand_unindex(&river_indexer, 3, i, cards))
                    return;
                
                std::bitset<52> deck;
                for (int i = 0; i < 7; ++i)
                {
                    deck.flip(cards[i]);
                }

                privateCards = {Card(cards[0]), Card(cards[1])};
                publicCards = {Card(cards[2]), Card(cards[3]), Card(cards[4]), Card(cards[5]), Card(cards[6])};
                riverEHS[i] = static_cast<uint16_t>(10000.0f * getEHS(privateCards, publicCards, deck));
            }
        };

        Thread::startThreadedLoop(callback, nCanonicalHands, nThreads);
    }
    
    // NOTE (keb): Opponent Cluster Hand Strength
    // https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf
    void generateRiverOCHS()
    {
        // 8 hand ranges
        // Histograms have 8 buckets
        // Each bucket quantity represents the EHS of the hand against that hand range
        using namespace PokerTypes;
        
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
        hand_indexer_t preflop_indexer;
        assert(hand_indexer_init(1, (const uint8_t[]){2}, &preflop_indexer));

        // NOTE (keb): Fill the Opponent Clusters with the appropriate hands.
        uint8_t cards[2];
        for (int card1 = 0; card1 < 52; ++card1)
        {
            for (int card2 = card1 + 1; card2 < 52; ++card2)
            {
                cards[0] = card1;
                cards[1] = card2;
                uint64_t index = hand_index_last(&preflop_indexer, cards);
                int ochsIndex = AbstractionsContext::preflopCanonicalIndexToOCHSIndex[index];
                AbstractionsContext::preflopOCHSBuckets[ochsIndex].push_back({card1, card2});
            }
        }

        uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, 3);
        auto riverOCHS = Memory::getMmap<Histogram<AbstractionsContext::nOCHSHistogramsBins>>(AbstractionsContext::riverOCHSFilename, nCanonicalHands, false);

        std::cout << std::endl;
        std::cout << "Generating River OCHS" << std::endl;
        auto callback = [&river_indexer, &riverOCHS](uint32_t start, uint32_t end, int threadid)
        {
            auto startChrono = std::chrono::high_resolution_clock::now();
            std::array<Card, 2> privateCards;
            std::array<Card, 5> publicCards;

            for (uint32_t i = start; i < end; ++i)
            {
                if (uint32_t done = i-start; done % 1'000'000 == 0 && threadid == 0)
                {
                    std::cout << "Time Remaining: " << Time::timeRemaining(end-start, i-start, startChrono) << " (s)" << std::endl;
                }

                uint8_t cards[7];
                if (!hand_unindex(&river_indexer, 3, i, cards))
                {
                    std::cerr << "Failed to unindex canonical index." << std::endl;
                    return;
                }
                
                std::bitset<52> deck;
                for (int i = 0; i < 7; ++i)
                {
                    deck.flip(cards[i]);
                }

                privateCards = {Card(cards[0]), Card(cards[1])};
                publicCards = {Card(cards[2]), Card(cards[3]), Card(cards[4]), Card(cards[5])};
                riverOCHS[i] = getOCHS(privateCards, publicCards, deck);
            }
        };

        Thread::startThreadedLoop(callback, nCanonicalHands, 12);
    }

    // NOTE (keb): no abstraction preflop. 
    void createRoundsAbstractions()
    {
        generateRiverEHS();
        generateRoundsHistograms();
        generateRiverOCHS();
        writeRoundsCentroids();
    }



}; // namespace EHS