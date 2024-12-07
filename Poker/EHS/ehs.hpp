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

    namespace AbstractionsContext
    {
        const char* riverEHSFilename = "ehs.dat";
        const char* riverOCHSFilename = "riverOchs.dat";
        const char* flopHistogramsFilename = "flopHistograms.dat";
        const char* turnHistogramsFilename = "turnHistograms.dat";
        const char* flopCentroidsFilename = "flopCentroids.dat";
        const char* turnCentroidsFilename = "turnCentroids.dat"; 

        constexpr int nBuckets = 200;
        constexpr int nEHSHistogramsBins = 50;
        constexpr int nOCHSHistogramsBins = 8;
        
        const std::unordered_map<uint64_t, int> preflopCanonicalIndexToOCHSIndex
        {
            {0, 3},    // 22
            {1, 0},    // 23o
            {2, 5},    // 33
            {3, 0},    // 24o
            {4, 0},    // 43o
            {5, 5},    // 44
            {6, 0},    // 52o
            {7, 0},    // 53o
            {8, 0},    // 54o
            {9, 5},    // 55
            {10, 0},   // 62o
            {11, 0},   // 63o
            {12, 0},   // 64o
            {13, 0},   // 65o
            {14, 6},   // 66
            {15, 0},   // 72o
            {16, 0},   // 73o
            {17, 0},   // 74o
            {18, 1},   // 75o
            {19, 2},   // 76o
            {20, 6},   // 77
            {21, 0},   // 82o
            {22, 0},   // 83o
            {23, 1},   // 84o
            {24, 1},   // 85o
            {25, 2},   // 86o
            {26, 2},   // 87o
            {27, 7},   // 88
            {28, 1},   // 92o
            {29, 1},   // 93o
            {30, 1},   // 94o
            {31, 1},   // 95o
            {32, 2},   // 96o
            {33, 2},   // 97o
            {34, 2},   // 98o
            {35, 7},   // 99
            {36, 1},   // T2o
            {37, 1},   // T3o
            {38, 1},   // T4o
            {39, 1},   // T5o
            {40, 2},   // T6o
            {41, 2},   // T7o
            {42, 2},   // T8o
            {43, 4},   // T9o
            {44, 7},   // TT
            {45, 1},   // J2o
            {46, 1},   // J3o
            {47, 3},   // J4o
            {48, 3},   // J5o
            {49, 3},   // J6o
            {50, 3},   // J7o
            {51, 4},   // J8o
            {52, 4},   // J9o
            {53, 4},   // JTo
            {54, 7},   // JJ
            {55, 3},   // Q2o
            {56, 3},   // Q3o
            {57, 3},   // Q4o
            {58, 3},   // Q5o
            {59, 3},   // Q6o
            {60, 3},   // Q7o
            {61, 4},   // Q8o
            {62, 4},   // Q9o
            {63, 4},   // QTo
            {64, 4},   // QJo
            {65, 7},   // QQ
            {66, 3},   // K2o
            {67, 3},   // K3o
            {68, 3},   // K4o
            {69, 5},   // K5o
            {70, 5},   // K6o
            {71, 5},   // K7o
            {72, 5},   // K8o
            {73, 5},   // K9o
            {74, 6},   // KTo
            {75, 6},   // KJo
            {76, 6},   // KQo
            {77, 7},   // KK
            {78, 5},   // A2o
            {79, 5},   // A3o
            {80, 5},   // A4o
            {81, 5},   // A5o
            {82, 5},   // A6o
            {83, 5},   // A7o
            {84, 5},   // A8o
            {85, 6},   // A9o
            {86, 6},   // ATo
            {87, 6},   // AJo
            {88, 6},   // AQo
            {89, 6},   // AKo
            {90, 7},   // AA
            {91, 0},   // 23s
            {92, 0},   // 24s
            {93, 0},   // 34s
            {94, 0},   // 25s
            {95, 0},   // 35s
            {96, 0},   // 45s
            {97, 0},   // 26s
            {98, 0},   // 36s
            {99, 0},   // 46s
            {100, 2},  // 56s
            {101, 0},  // 27s
            {102, 0},  // 37s
            {103, 1},  // 47s
            {104, 2},  // 57s
            {105, 2},  // 67s
            {106, 1},  // 28s
            {107, 1},  // 38s
            {108, 1},  // 48s
            {109, 2},  // 58s
            {110, 2},  // 68s
            {111, 2},  // 78s
            {112, 1},  // 29s
            {113, 1},  // 39s
            {114, 1},  // 49s
            {115, 2},  // 59s
            {116, 2},  // 69s
            {117, 2},  // 79s
            {118, 2},  // 89s
            {119, 1},  // 2Ts
            {120, 2},  // 3Ts
            {121, 2},  // 4Ts
            {122, 2},  // 5Ts
            {123, 2},  // 6Ts
            {124, 4},  // 7Ts
            {125, 4},  // 8Ts
            {126, 4},  // 9Ts
            {127, 3},  // 2Js
            {128, 3},  // 3Js
            {129, 3},  // 4Js
            {130, 3},  // 5Js
            {131, 3},  // 6Js
            {132, 4},  // 7Js
            {133, 4},  // 8Js
            {134, 4},  // 9Js
            {135, 4},  // TJs
            {136, 3},  // 2Qs
            {137, 3},  // 3Qs
            {138, 3},  // 4Qs
            {139, 3},  // 5Qs
            {140, 4},  // 6Qs
            {141, 4},  // 7Qs
            {142, 4},  // 8Qs
            {143, 4},  // 9Qs
            {144, 6},  // TQs
            {145, 6},  // JQs
            {146, 3},  // 2Ks
            {147, 5},  // 3Ks
            {148, 5},  // 4Ks
            {149, 5},  // 5Ks
            {150, 5},  // 6Ks
            {151, 5},  // 7Ks
            {152, 5},  // 8Ks
            {153, 6},  // 9Ks
            {154, 6},  // TKs
            {155, 6},  // JKs
            {156, 6},  // QKs
            {157, 5},  // 2As
            {158, 5},  // 3As
            {159, 5},  // 4As
            {160, 5},  // 5As
            {161, 5},  // 6As
            {162, 6},  // 7As
            {163, 6},  // 8As
            {164, 6},  // 9As
            {165, 6},  // TAs
            {166, 6},  // JAs
            {167, 6},  // QAs
            {168, 6}   // KAs
        };
    };

    struct HandStats
    {
        std::atomic<uint32_t> winner {0};
        std::atomic<uint32_t> loser {0};
        std::atomic<uint32_t> draw {0};
        float pctWinner {0};
        float pctLoser {0};
        float pctDraw {0};

        void updatePct()
        {
            uint64_t sum = winner + loser + draw;
            pctWinner = winner / static_cast<double>(sum);
            pctLoser = loser / static_cast<double>(sum);
            pctDraw = draw / static_cast<double>(sum);
        }
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
    
    inline Histogram<AbstractionsContext::nOCHSHistogramsBins> getOCHS(const std::array<PokerTypes::Card, 2>& myCards, const std::array<PokerTypes::Card, 5>& boardCards, const std::bitset<52>& deck)
    {
        using namespace PokerTypes;
        HandStats stats;
        int myRank = evaluate_7cards(myCards[0], myCards[1], boardCards[0], boardCards[1], boardCards[2], boardCards[3], boardCards[4]);

        Histogram<AbstractionsContext::nOCHSHistogramsBins> result;

        for (int i = 0; i < 8; ++i)
        {
            // TODO: getEHS against a specific HAND RANGE.
            // result[i] = getEHS()
        }
        // for (int o1 = 0; o1 < 52; ++o1)
        // {
        //     if (deck.test(o1)) [[unlikely]]
        //         continue;
        //     for (int o2 = o1+1; o2 < 52; ++o2)
        //     {
        //         if (deck.test(o2)) [[unlikely]]
        //             continue;

        //         int theirRank = evaluate_7cards(o1, o2, boardCards[0], boardCards[1], boardCards[2], boardCards[3], boardCards[4]);

        //         myRank < theirRank ? ++stats.winner : myRank > theirRank ? ++stats.loser : ++stats.draw;
        //     }
        // }

        // float result = (stats.winner + (stats.draw / 2.0)) / (stats.winner + stats.loser + stats.draw);
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
        
        constexpr int nThreads = 12; 
        uint32_t blockSize = nCanonicalHandsTurn / nThreads;
        uint32_t leftover = nCanonicalHandsTurn % nThreads;
        std::vector<std::thread> threads;
        for (int i = 0; i < nThreads; ++i)
        {
            if (i == nThreads-1)
            {
                threads.emplace_back(callback, i*blockSize, i*blockSize+blockSize+leftover, i);
                break;
            }
            threads.emplace_back(callback, i*blockSize, i*blockSize+blockSize, i);
        }

        for_each(threads.begin(), threads.end(), [](auto& thread){ thread.join(); });
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

    void writeRoundCentroids(const char* centroidsFilename, 
        const char* histogramFilename,
        int nBuckets, 
        int roundIndex)
    {
        std::cout << std::endl;
        std::cout << "Computing and writing centroids for " << centroidsFilename << std::endl;
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));
        uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, roundIndex);
        auto histograms = Memory::getMmap<Histogram<AbstractionsContext::nEHSHistogramsBins>>(histogramFilename, nCanonicalHands);


        std::vector<Histogram<AbstractionsContext::nEHSHistogramsBins>> result (nBuckets);
        std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> dataPoints (histograms.get(), histograms.get() + nCanonicalHands);
        result = KMeans::kMeansClustering<Histogram<AbstractionsContext::nEHSHistogramsBins>, 
            KMeans::EMDDistance<Histogram<AbstractionsContext::nEHSHistogramsBins>>, 
            KMeans::PipePipeCentroidsInitializer<Histogram<AbstractionsContext::nEHSHistogramsBins>, KMeans::EMDDistance<Histogram<AbstractionsContext::nEHSHistogramsBins>>, 0, std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>>>
            > (nBuckets, dataPoints);

        
        auto centroids = Memory::getMmap<Histogram<AbstractionsContext::nEHSHistogramsBins>>(centroidsFilename, nBuckets, false);
        for (int i = 0; i < nBuckets; ++i)
        {
            centroids[i] = std::move(result[i]);
        }
    }

    void writeRoundsCentroids()
    {
        writeRoundCentroids(AbstractionsContext::flopCentroidsFilename,
            AbstractionsContext::flopHistogramsFilename, 
            AbstractionsContext::nBuckets, 1); 
        
        writeRoundCentroids(AbstractionsContext::turnCentroidsFilename,
            AbstractionsContext::turnHistogramsFilename, 
            AbstractionsContext::nBuckets, 2);
    }

    void createRoundsAbstractions()
    {
        generateRoundsHistograms();
        writeRoundsCentroids();
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
            uint32_t total = end-start;
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

        Thread::startThreadedLoop(callback, nCanonicalHands, 12);
    }

    void generateRiverOCHS()
    {
        // 8 hand ranges
        // Histograms have 8 buckets
        // Each bucket quantity represents the EHS of the hand against that hand range
        using namespace PokerTypes;
        
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));

        uint64_t nCanonicalHands = hand_indexer_size(&river_indexer, 3);
        auto riverOCHS = Memory::getMmap<Histogram<AbstractionsContext::nOCHSHistogramsBins>>(AbstractionsContext::riverOCHSFilename, nCanonicalHands, false);

        std::cout << std::endl;
        std::cout << "Generating River OCHS" << std::endl;
        auto callback = [&river_indexer, &riverOCHS](uint32_t start, uint32_t end, int threadid)
        {
            auto startChrono = std::chrono::high_resolution_clock::now();
            uint32_t total = end-start;
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


}; // namespace EHS