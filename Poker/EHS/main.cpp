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
        
    std::array<hand_indexer_t, 4> canonical_indexers;
    assert(hand_indexer_init(1, (const uint8_t[]){2}, &canonical_indexers[0]));
    assert(hand_indexer_init(2, (const uint8_t[]){2,3}, &canonical_indexers[1]));
    assert(hand_indexer_init(3, (const uint8_t[]){2,3,1}, &canonical_indexers[2]));
    assert(hand_indexer_init(4, (const uint8_t[]){2,3,1,1}, &canonical_indexers[3]));

    constexpr size_t HugePageSize = 2 * 1024 * 1024;
    constexpr uint64_t _7C2 = 21;
    uint64_t nCanonicalHands = canonical_indexers[3].round_size[3];
    uint64_t numElements = nCanonicalHands*_7C2;
    size_t fileSizeEHS = numElements * sizeof(uint16_t);

    FILE* fdEHS = fopen("ehs.dat", "w+x");
    if (fseek(fdEHS, fileSizeEHS-1, SEEK_SET) != 0)
    {
        std::cerr << "fseek failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    if (fwrite("\0", 1, 1, fdEHS) < 1)
    {
        std::cerr << "fwrite failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    uint16_t* riverEHS = (uint16_t*)mmap(NULL, fileSizeEHS, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fileno(fdEHS), 0);
    fclose(fdEHS);


    if (riverEHS == MAP_FAILED) {
        std::cerr << "Memory allocation failed!" << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    EHS::generateEHSRiverCanonical(riverEHS);


    // ======================= TURN HISTOGRAMS =========================== 
    constexpr uint64_t _6C2 = 15;
    nCanonicalHands = canonical_indexers[2].round_size[2];
    numElements = nCanonicalHands*_6C2;
    size_t fileSizeTurn = numElements * sizeof(PokerTypes::Histogram);
    
    FILE* fdTurn = fopen("turnHistograms.dat", "w+x");
    if (fseek(fdTurn, fileSizeTurn-1, SEEK_SET) != 0)
    {
        std::cerr << "fseek failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    if (fwrite("\0", 1, 1, fdTurn) < 1)
    {
        std::cerr << "fwrite failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    Histogram* turnHistograms = (Histogram*)mmap(NULL, fileSizeTurn, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fileno(fdTurn), 0);
    fclose(fdTurn);

    if (turnHistograms == MAP_FAILED) {
        std::cerr << "Memory allocation failed!" << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    EHS::generateTurnHistograms(PokerTypes::BettingRound::Turn, riverEHS, turnHistograms);

    EHS::getTurnBuckets(200, turnHistograms);
    
    // ======================= FLOP HISTOGRAMS =========================== 
    constexpr uint64_t _5C2 = 10;
    nCanonicalHands = canonical_indexers[1].round_size[1];
    numElements = nCanonicalHands*_5C2;
    size_t fileSizeFlop = numElements * sizeof(PokerTypes::Histogram);
    
    FILE* fdFlop = fopen("flopHistograms.dat", "w+x");
    if (fseek(fdFlop, fileSizeFlop-1, SEEK_SET) != 0)
    {
        std::cerr << "fseek failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    if (fwrite("\0", 1, 1, fdFlop) < 1)
    {
        std::cerr << "fwrite failed." << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }

    Histogram* flopHistograms = (Histogram*)mmap(NULL, fileSizeFlop, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fileno(fdFlop), 0);
    fclose(fdFlop);

    if (flopHistograms == MAP_FAILED) {
        std::cerr << "Memory allocation failed!" << std::endl;
        std::cerr << errno << std::endl;
        return 1;
    }
    
    EHS::generateFlopHistograms(PokerTypes::BettingRound::Flop, turnHistograms, flopHistograms);

    EHS::getFlopBuckets(200, flopHistograms);

    // ================ FREE ===================
    munmap(flopHistograms, fileSizeFlop);
    munmap(turnHistograms, fileSizeTurn);
}