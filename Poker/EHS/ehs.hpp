#pragma once

#include <pokerTypes.hpp>

#include <kmeans.hpp>

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
#include <fstream>
#include <iostream> // TODO: remove after debugging
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

    std::array<uint8_t, 256> privateCardsIndexFast;
    // TODO (keb): redo this so you can have a common mapping for every betting round.
    std::unordered_map<uint8_t, uint32_t> privateCardsIndexTurn
    {
        {0b000011,0},
        {0b000101,1},
        {0b001001,2},
        {0b010001,3},
        {0b100001,4},

        {0b0000110,5},
        {0b0001010,6},
        {0b0010010,7},
        {0b0100010,8},
        
        {0b001100,9},
        {0b010100,10},
        {0b100100,11},
        
        {0b011000,12},
        {0b101000,13},
        
        {0b110000,14},
    };
    std::unordered_map<uint8_t, uint32_t> privateCardsIndexRiver
    {
        {0b0000011,0},
        {0b0000101,1},
        {0b0001001,2},
        {0b0010001,3},
        {0b0100001,4},
        {0b1000001,5},

        {0b0000110,6},
        {0b0001010,7},
        {0b0010010,8},
        {0b0100010,9},
        {0b1000010,10},
        
        {0b0001100,11},
        {0b0010100,12},
        {0b0100100,13},
        {0b1000100,14},
        
        {0b0011000,15},
        {0b0101000,16},
        {0b1001000,17},
        
        {0b0110000,18},
        {0b1010000,19},
        
        {0b1100000,20}
    };

    int nCr(int n, int r)
    {
        //    n!
        // --------
        // (n-r)!r!
        int numerator = 1;
        for (int i = n; i > n - (n-r); --i)
        {
            numerator *= i;
        }

        return numerator / factorial(r, r);
    }

    template <typename E>
    int enumToInt(E e)
    {
        return static_cast<int>(e);
    }

    using Histogram = std::array<uint8_t, 20>;

    std::vector<Histogram> getFlopBuckets(int nBuckets, Histogram* flopHistograms)
    {
        std::vector<Histogram> result (nBuckets);
        
        hand_indexer_t canonical_indexer;
        assert(hand_indexer_init(2, (const uint8_t[]){2,3}, &canonical_indexer));

        std::span<Histogram> dataPoints (flopHistograms, flopHistograms + (canonical_indexer.round_size[1] * nCr(5,2)));
        KMeans::kMeansClustering<Histogram, 
            KMeans::EMDDistance<Histogram>, 
            KMeans::PlusPlusCentroidsInitializer<Histogram, KMeans::EMDDistance<Histogram>, 42>
            > (nBuckets, dataPoints);


        return result;
    }

    void generateFlopHistograms(PokerTypes::BettingRound round, Histogram* turnHistograms, Histogram* flopHistograms)
    {
        std::array<hand_indexer_t, 2> canonical_indexers;
        assert(hand_indexer_init(2, (const uint8_t[]){2,3}, &canonical_indexers[0]));
        assert(hand_indexer_init(3, (const uint8_t[]){2,3,1}, &canonical_indexers[1]));

        int nPrivateHandsPerCanonical = nCr(5, 2);
        int nPrivateHandsPerCanonicalTurn = nCr(6, 2);
        assert(nPrivateHandsPerCanonical == 10);
        
        for (int canonicalIndex = 0; 
            canonicalIndex < canonical_indexers[0].round_size[enumToInt(round)]; 
            ++canonicalIndex)
        {
            uint8_t cards[5];
            if (!hand_unindex(&canonical_indexers[0], 1, canonicalIndex, cards))
            {
                return;
            }

            std::bitset<52> deck;
            for (int i = 0; i < 5; ++i)
            {
                deck.flip(cards[i]);
            }

            int intraCanonicalOffset = 0;
            for (int p1 = 0; p1 < 5; ++p1)
            {
                for (int p2 = p1+1; p2 < 5; ++p2)
                {
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
                        int turnIndex = hand_index_last(&canonical_indexers[1], turnCards);
                        std::array<int, 2> privateCardsTurnIndex;
                        int index = 0;
                        for (int i = 0; i < 6; ++i)
                        {
                            if (turnCards[i] == cards[p1] || turnCards[i] == cards[p2])
                            {
                                privateCardsTurnIndex[index++] = i;
                            }
                        }
                        Histogram turnHistogram = turnHistograms[turnIndex*nPrivateHandsPerCanonicalTurn+privateCardsIndexTurn[1 << privateCardsTurnIndex[0] | 1 << privateCardsTurnIndex[1]]];
                        for (int i = 0; i < turnHistogram.size(); ++i)
                        {
                            histogram[i] += turnHistogram[i];
                        }
                    }
                    flopHistograms[canonicalIndex*nPrivateHandsPerCanonical+intraCanonicalOffset++] = histogram;
                }
            }
        }
    }


    // NOTE (keb): a histogram keeps track of the number of times a hand ends up
    // with a given strength on the river. The bins intervals represent the strength.
    // The bins values represent the number of time at that strength.
    void generateTurnHistograms(PokerTypes::BettingRound round, uint16_t* riverEHS, Histogram* turnHistograms)
    {
        std::array<hand_indexer_t, 2> canonical_indexers;
        assert(hand_indexer_init(3, (const uint8_t[]){2,3,1}, &canonical_indexers[0]));
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &canonical_indexers[1]));

        int nPrivateHandsPerCanonical = nCr(6, 2);
        int nPrivateHandsPerCanonicalRiver = nCr(7, 2);
        assert(nPrivateHandsPerCanonical == 15);

        for (int canonicalIndex = 0; 
            canonicalIndex < canonical_indexers[0].round_size[enumToInt(round)]; 
            ++canonicalIndex)
        {
           uint8_t cards[6];
           if (!hand_unindex(&canonical_indexers[0], 2, canonicalIndex, cards))
           {
              return;
           }

           std::bitset<52> deck;
           for (int i = 0; i < 6; ++i)
           {
            deck.flip(cards[i]);
           }
           
           int intraCanonicalOffset = 0;
           for (int p1 = 0; p1 < 6; ++p1)
           {
            for (int p2 = p1+1; p2 < 6; ++p2)
            {
                Histogram histogram; 
                for (int i = 0; i < 52; ++i)
                {
                    if (deck.test(i))
                    {
                        continue;
                    }
                    uint8_t riverCards[7];
                    std::memcpy(riverCards, cards, 6);
                    riverCards[6] = i;
                    int riverIndex = hand_index_last(&canonical_indexers[1], riverCards);
                    std::array<int, 2> privateCardsRiverIndex;
                    int index = 0;
                    for (int i = 0; i < 7; ++i)
                    {
                        if (riverCards[i] == cards[p1] || riverCards[i] == cards[p2])
                        {
                            privateCardsRiverIndex[index++] = i;
                        }
                    }
                    uint16_t riverStrength = riverEHS[riverIndex*nPrivateHandsPerCanonicalRiver+privateCardsIndexRiver[1 << privateCardsRiverIndex[0] | 1 << privateCardsRiverIndex[1]]];
                    double rhs = riverStrength / 10'000;
                    // NOTE (keb): histogram has 20 bins
                    // 0.05 0.10 0.15 0.20 0.25 0.30 0.35 ... 0.95 1.00
                    // [0,0.05), [0.05, 0.10), ..., [0.95, 1.00]
                    int binIndex = rhs*histogram.size();
                    ++histogram[binIndex == 20 ? 19: binIndex];
                }
                turnHistograms[canonicalIndex*nPrivateHandsPerCanonical+intraCanonicalOffset++] = histogram;
            }
           } 
        }
    }

    void generateEHSRiverCanonical(uint16_t* riverEHS)
    {
        using namespace PokerTypes;
        hand_indexer_t river_indexer;
        assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &river_indexer));

        auto callback = [&river_indexer, &riverEHS](uint32_t start, uint32_t end, int threadid)
        {
            auto startChrono = std::chrono::high_resolution_clock::now();
            uint32_t total = end-start;
            std::cout << std::fixed << std::setprecision(2);
            std::array<Card, 2> privateCards;
            std::array<Card, 5> publicCards;

            for (uint32_t i = start; i < end; ++i)
            {
                if (uint32_t done = i-start; done % 1'000'000 == 0)
                {
                    std::cout << threadid << ": " << (done/(double)total) * 100 << "%" << std::endl;
                    auto endChrono = std::chrono::high_resolution_clock::now();
                    auto timeElapsed = std::chrono::duration<double>(endChrono-startChrono).count();
                    std::cout << "Time Elapsed: " << timeElapsed << " (s)" << std::endl;
                    std::cout << "Time Remaining: " << (timeElapsed*total)/done << " (s)" << std::endl;
                }

                uint8_t cards[7];
                if (!hand_unindex(&river_indexer, 3, i, cards))
                    return;
                
                std::bitset<52> deck;
                for (int i = 0; i < 7; ++i)
                {
                    deck.flip(cards[i]);
                }

                int intraCanonicalOffset = 0;
                for (int p1 = 0; p1 < 7; ++p1)
                {
                    for (int p2 = p1+1; p2 < 7; ++p2)
                    {
                        privateCards = {Card(cards[p1]), Card(cards[p2])};
                        int index = 0;
                        for (int c = 0; c < 7; ++c)
                        {
                            if (c != p1 && c != p2)
                            {
                                publicCards[index++] = Card(cards[c]);
                            }
                        }
                        riverEHS[i*21+intraCanonicalOffset] = static_cast<uint16_t>(10000.0f * getEHS(privateCards, publicCards, deck));
                        ++intraCanonicalOffset;
                    }
                }
            }
        };

        constexpr int nThreads = 12; 
        uint32_t blockSize = river_indexer.round_size[3] / nThreads;
        uint32_t leftover = river_indexer.round_size[3] % nThreads;
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


}; // namespace EHS