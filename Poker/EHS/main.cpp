#include <ehs.hpp>

extern "C"
{
    #include <hand_index.h>
}

#include <pokerTypes.hpp>

#include <chrono>
#include <cstring>
#include <iostream>

#include <sys/mman.h>
#include <errno.h>

int main()
{
    using namespace PokerTypes;


    // EHS::generateFlopHistograms();
        
    hand_indexer_t turn_indexer;
    assert(hand_indexer_init(3, (const uint8_t[]){2,3,1}, &turn_indexer));
    uint64_t nCanonicalHandsTurn = hand_indexer_size(&turn_indexer, 2);
    auto histogramsTurn = Memory::getMmap<Histogram>("turnHistograms.dat", nCanonicalHandsTurn);
    
    // hand_indexer_t flop_indexer;
    // assert(hand_indexer_init(2, (const uint8_t[]){2,3}, &flop_indexer));
    // uint64_t nCanonicalHandsFlop = hand_indexer_size(&flop_indexer, 1);
    // auto histogramsFlop = Memory::getMmap<Histogram>("flopHistograms.dat", nCanonicalHandsFlop);

    // uint8_t cards[] = {0, 29, 5, 17, 40, 41};
    // auto ti = hand_index_last(&turn_indexer, cards);
    // Histogram& h = histograms[ti];
    // std::cout << h.getPercent().toString() << std::endl;
    
    // uint8_t cards[] = {49, 50, 51, 17, 40};
    // auto fi = hand_index_last(&flop_indexer, cards);
    // Histogram& h = histogramsFlop[fi];
    // std::cout << h.getPercent().toString() << std::endl;

    EHS::writeRoundCentroids("turnHistogramsCentroids.dat", 200, histogramsTurn, BettingRound::Turn, turn_indexer); 
}