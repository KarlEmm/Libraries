#pragma once

#include <random.hpp>

#include <array>
#include <cassert>
#include <cstdint>
#include <string>

namespace PokerTypes 
{

namespace AbstractionsContext
{
	const char* riverEHSFilename = "ehs.dat";
	const char* riverOCHSFilename = "riverOchs.dat";
	const char* flopHistogramsFilename = "flopHistograms.dat";
	const char* turnHistogramsFilename = "turnHistograms.dat";
	const char* flopCentroidsFilename = "flopCentroids.dat";
	const char* turnCentroidsFilename = "turnCentroids.dat"; 
	const char* riverCentroidsFilename = "riverCentroids.dat"; 

	constexpr int nFlopBuckets = 200;
	constexpr int nTurnBuckets = 200;
	constexpr int nRiverBuckets = 200;
	constexpr int nEHSHistogramsBins = 50;
	constexpr int nOCHSHistogramsBins = 8;

	std::vector<std::vector<std::pair<uint8_t, uint8_t>>> preflopOCHSBuckets (nOCHSHistogramsBins, std::vector<std::pair<uint8_t, uint8_t>>()); 
	std::unordered_map<uint64_t, int> preflopCanonicalIndexToOCHSIndex
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


template <int N = 50>
struct Histogram
{
	using Data = std::array<float, N>;

	Histogram() = default;
	Histogram(const Histogram&) = default;
	Histogram(Histogram&&) = default;
	Histogram& operator=(const Histogram&) = default;
	Histogram& operator=(Histogram&&) = default;

	Histogram& operator+=(const Histogram& other)
	{
        auto thisItr = m_data.begin();
        auto otherItr = other.m_data.begin();

        for (; thisItr != m_data.end() && otherItr != other.m_data.end(); ++thisItr, ++otherItr)
        {
            *thisItr += *otherItr;
        }

		m_nElementsStored += other.m_nElementsStored; 
        return *this;
	}
    
	Histogram operator/(double divisor)
    {
        Histogram result;
		for (int i = 0; i < m_data.size(); ++i)
		{
            result.m_data[i] = (m_data[i] / divisor);
        }
        return result;
    }
	
	Histogram& operator/=(double divisor)
    {
		for (int i = 0; i < m_data.size(); ++i)
		{
            m_data[i] /= divisor;
        }
        return *this;
    }

	Histogram& operator*=(double factor)
    {
        for (auto& e : m_data)
        {
            e *= factor;
        }
        return *this;
    }

	Histogram& convertToPercent()
	{
		this->operator/=(m_nElementsStored);
		return *this;
	}
	
	Histogram getPercent()
	{
		return *this / m_nElementsStored;
	}

	std::string toString()
	{
		std::string result;
		for (int i = 0; i < m_data.size(); ++i)
		{
			result += (i == 0 ? std::to_string(m_data[i]) : '|' + std::to_string(m_data[i]));
		}
		result += '\n';
		return result;
	}

	void incrementBin(size_t binIndex, int n = 1)
	{
		m_data[binIndex] += n;
		m_nElementsStored += n;
	}

	size_t size() const { return m_data.size(); }
	float& operator[](size_t index) { return m_data[index]; }
	const float& operator[](size_t index) const { return m_data[index]; }
    
	Data::iterator begin() { return m_data.begin(); }
	Data::iterator end() { return m_data.end(); }
	Data::const_iterator begin() const { return m_data.begin(); }
	Data::const_iterator end() const { return m_data.end(); }

	Data m_data {0};
	uint32_t m_nElementsStored {0};
	bool isConvertedToPercent {false};
};

enum Action : uint8_t
{
	Fold,
	Call,
	Check,
	BetQuarter,
	BetHalf,
	BetPot,
	BetOver,
	Count
};

enum class BettingRound : uint8_t
{
	Preflop,
	Flop,
	Turn,
	River,
	RoundCount
};

// NOTE (keb): 6 is the most actions that can happen on a heads up betting round.
using GameHistory = std::array<std::array<Action, 6>, (uint8_t)BettingRound::RoundCount>;

namespace Constants
{
	constexpr int strategyInterval{ 10'000 };
	constexpr int pruningThresholdMinutes{ 200 };
	constexpr int LCFRThresholdMinutes{ 400 };
	constexpr int discountIntervalMinutes{ 10 };
	
	constexpr int maxStartingStackSize{ 800 };
	constexpr int minStartingStackSize{ 200 };
	
	constexpr int sb{ 10 };
	constexpr int bb{ sb * 2 };
	
	constexpr int nPlayers{ 2 };
	constexpr int nFlopCards{ 3 };
	constexpr int nTurnCards{ 1 };
	constexpr int nRiverCards{ 1 };
	constexpr int nCommunityCards{ nFlopCards + nTurnCards + nRiverCards };
	constexpr int nPrivateCards{ 2 };
    constexpr int nTotalPlayerCards = nPrivateCards * nPlayers;
	constexpr int nTotalCards = nTotalPlayerCards + nCommunityCards;
	constexpr std::array<int, (uint8_t)BettingRound::RoundCount> nCardsRoundAccumulator = {
		nPrivateCards, 
		nPrivateCards + nFlopCards, 
		nPrivateCards + nFlopCards + nTurnCards,
		nPrivateCards + nFlopCards + nTurnCards + nRiverCards
	};
	constexpr int communityCardIndex = nPrivateCards * nPlayers;

	constexpr int nRanks{ 13 };
	constexpr int nSuits{ 4 };
	constexpr int nDeckCards{ nRanks * nSuits };
	const char RANK_TO_CHAR[] = "23456789TJQKA";
	const char SUIT_TO_CHAR[] = "shdc";
};


struct Card
{
	Card() : rank{ 0 }, suit{ 0 } {};
	explicit Card(uint8_t card) : rank{ static_cast<uint8_t>(card >> 2) }, suit{ static_cast<uint8_t>(card & 0b11) } {};
	uint8_t rank : 6;
	uint8_t suit : 2;

    std::string to_string() 
	{
		std::string result = std::string{static_cast<char>(Constants::RANK_TO_CHAR[rank]), static_cast<char>(Constants::SUIT_TO_CHAR[suit])}; 
		return result;
	}

	operator int() { return (rank << 2) | (suit); }
	operator int() const { return (rank << 2) | (suit); }
};

struct Deck
{
    Deck()
    {
        init();
    }

	void init()
	{
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			deck[i] = Card{ static_cast<uint8_t>(i)};
		}
	}

	void shuffle()
	{
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			int rand = Random::get(0, i);
			std::swap(deck[i], deck[rand]);
		}
	}

    void randomizePlayerCards(int playerIndex)
    {
        for (int playerCardIndex = playerIndex * 2; playerCardIndex <= playerIndex * 2 + 1; ++playerCardIndex)
        {
			int rand = Random::get(firstUnusedCardIndex(), Constants::nDeckCards);
			std::swap(deck[playerCardIndex], deck[rand]);
        }
    }
    
    int firstUnusedCardIndex()
    {
        return Constants::nTotalCards;
    }

    void setPlayerCards(int playerIndex, const std::array<Card, Constants::nPrivateCards>& cards)
    {
        std::swap(deck[playerIndex * 2], deck[static_cast<const int>(cards[0])]);
        std::swap(deck[playerIndex * 2 + 1], deck[static_cast<const int>(cards[1])]);
    }
    
    void setCommunityCards(const std::array<Card, Constants::nCommunityCards>& cards)
    {
        for (int i = 0; i < Constants::nCommunityCards; ++i)
        {
            std::swap(deck[Constants::nTotalPlayerCards + i], deck[static_cast<const int>(cards[i])]);
        }
    }
	
	std::array<Card, Constants::nPrivateCards> getPreflopHand(int playerIndex) const { return { deck[playerIndex * 2], deck[playerIndex * 2 + 1] }; } 
	std::array<Card, Constants::nFlopCards> getFlopCards() const { return { deck[Constants::communityCardIndex], deck[Constants::communityCardIndex+1], deck[Constants::communityCardIndex+2] }; } 
	std::array<Card, Constants::nTurnCards> getTurnCards() const { return { deck[Constants::communityCardIndex + 3] }; } 
	std::array<Card, Constants::nRiverCards> getRiverCards() const { return { deck[Constants::communityCardIndex + 4] }; } 
	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> getFlopHand(int playerIndex) const
	{
		const auto& preflopCards = getPreflopHand(playerIndex);
		const auto& flopCards = getFlopCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> flopHand;
		std::copy(preflopCards.begin(), preflopCards.end(), flopHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), flopHand.begin() + Constants::nPrivateCards);

		return flopHand;
	}
	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> getTurnHand(int playerIndex) const
	{
		const auto& preflopCards = getPreflopHand(playerIndex);
		const auto& flopCards = getFlopCards();
		const auto& turnCards = getTurnCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> turnHand;
		std::copy(preflopCards.begin(), preflopCards.end(), turnHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), turnHand.begin() + Constants::nPrivateCards);
		std::copy(turnCards.begin(), turnCards.end(), turnHand.begin() + Constants::nPrivateCards + Constants::nFlopCards);

		return turnHand;
	}
	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards + Constants::nRiverCards> getRiverHand(int playerIndex) const
	{
		const auto& preflopCards = getPreflopHand(playerIndex);
		const auto& flopCards = getFlopCards();
		const auto& turnCards = getTurnCards();
		const auto& riverCards = getRiverCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards + Constants::nRiverCards> riverHand;
		std::copy(preflopCards.begin(), preflopCards.end(), riverHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), riverHand.begin() + Constants::nPrivateCards);
		std::copy(turnCards.begin(), turnCards.end(), riverHand.begin() + Constants::nPrivateCards + Constants::nFlopCards);
		std::copy(riverCards.begin(), riverCards.end(), riverHand.begin() + Constants::nPrivateCards + Constants::nFlopCards + Constants::nRiverCards);

		return riverHand;
	}

	Card operator[](int index) { return deck[index]; }

	std::array<Card, Constants::nDeckCards> deck{ };

    // Bit Representation
    uint64_t cards {0};

    bool isCardUsed(Card c) { return cards & (1 << static_cast<int>(c)); }

    // ==================
};

};