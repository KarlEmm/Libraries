// TODO (keb):
// 1. Write a Serializer to dump and read infosets from a file.
// 2. Write an interface to query the infosets.

#include <kmeans.hpp>
#include <memory.hpp>
#include <pokerTypes.hpp>
#include <random.hpp>
#include <time.hpp>

#include <phevaluator/phevaluator.h>
extern "C"
{
    #include <hand_index.h>
}

#include <array>
#include <cassert>
#include <chrono>
#include <iostream> // For debugging
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>


namespace MCCFR
{
using namespace PokerTypes;

std::array<hand_indexer_t, (uint8_t)BettingRound::RoundCount> indexers;

template <typename T>
concept SmallEnum = requires ()
{
	{ std::is_enum_v<T> };
	{ std::is_convertible_v<T, uint8_t> };
};

template <typename T>
requires SmallEnum<T> || (std::is_integral_v<T> && sizeof(T) <= 8)
struct Flags8Bits
{
	uint8_t flags { 0b0000'0000 };

	void set(T index) { flags |= 1 << index; }
	void unset(T index) { flags &= ~(1 << index); }
	bool test(T index) { return flags & (1 << index); }
	bool nSet() { return __builtin_popcount(flags); }
	bool one() { return nSet() == 1; }
	void reset() { flags = 0; }
};

std::string toString(Action action)
{
	if (action == Fold)
		return "F";
	if (action == Call)
		return "C";
	if (action == Check)
		return "CH";
	if (action == BetHalf)
		return "BH";
	if (action == BetPot)
		return "BP";
	if (action == BetOver)
		return "BO";
	return "?";
}

using Actions = Flags8Bits<Action>;

namespace Math
{
	bool isAlmostEqual(double a, double b, double epsilon = 0.00001)
	{
		return std::abs(a - b) < epsilon;
	}
};

namespace DebugGlobals
{
	int visited{ 0 };
	int nTrained {0};
};

struct Card
{
	Card() : rank{ 0 }, suit{ 0 } {};
	explicit Card(uint8_t card) : rank{ static_cast<uint8_t>(card >> 2) }, suit{ static_cast<uint8_t>(card & 0b11) } {};
	uint8_t rank : 6;
	uint8_t suit : 2;

	operator uint8_t() const {return (rank << 2) | (suit);}
	operator uint8_t() {return (rank << 2) | (suit);}

	operator int() { return (rank << 2) | (suit); }
};


struct Deck
{
	void init()
	{
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			deck[i] = Card{ static_cast<uint8_t>(i)};
		}
	}

	void shuffle()
	{
		init();
		for (int i = Constants::nDeckCards - 1; i >= 0; --i)
		{
			int rand = Random::get(0, i);
			std::swap(deck[i], deck[rand]);
		}
	}
	
	std::array<Card, Constants::nFlopCards> getFlopCards() const { return { deck[Constants::communityCardIndex], deck[Constants::communityCardIndex+1], deck[Constants::communityCardIndex+2] }; } 
	std::array<Card, Constants::nTurnCards> getTurnCards() const { return { deck[Constants::communityCardIndex + 3] }; } 
	std::array<Card, Constants::nRiverCards> getRiverCards() const { return { deck[Constants::communityCardIndex + 4] }; } 
	
	std::array<Card, Constants::nPrivateCards> getPreflopHand(int playerIdx) const { return { deck[playerIdx * 2], deck[playerIdx * 2 + 1] }; } 

	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> getFlopHand(int playerIdx) const
	{
		const auto& preflopCards = getPreflopHand(playerIdx);
		const auto& flopCards = getFlopCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards> flopHand;
		std::copy(preflopCards.begin(), preflopCards.end(), flopHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), flopHand.begin() + Constants::nPrivateCards);

		return flopHand;
	}

	std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> getTurnHand(int playerIdx) const
	{
		const auto& preflopCards = getPreflopHand(playerIdx);
		const auto& flopCards = getFlopCards();
		const auto& turnCards = getTurnCards();
		
		std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards> turnHand;
		std::copy(preflopCards.begin(), preflopCards.end(), turnHand.begin());
		std::copy(flopCards.begin(), flopCards.end(), turnHand.begin() + Constants::nPrivateCards);
		std::copy(turnCards.begin(), turnCards.end(), turnHand.begin() + Constants::nPrivateCards + Constants::nFlopCards);

		return turnHand;
	}

	using RiverHand = std::array<Card, Constants::nPrivateCards + Constants::nFlopCards + Constants::nTurnCards + Constants::nRiverCards>;
	RiverHand getRiverHand(int playerIdx) const
	{
		const auto& preflopCards = getPreflopHand(playerIdx);
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

	std::array<Card, Constants::nDeckCards> deck { };
};

using FoldedPlayers = Flags8Bits<uint8_t>;

inline int getPreviousPlayer(int currentPlayerId)
{
	return currentPlayerId == 0 ? Constants::nPlayers - 1 : currentPlayerId - 1;
}

inline int getNextPlayer(int currentPlayerId)
{
	return (currentPlayerId + 1) % Constants::nPlayers;
}

inline int getSBPlayerIdFromButton(int buttonPlayerId)
{
	return (buttonPlayerId + 1) % Constants::nPlayers;
}

inline int getBBPlayerIdFromButton(int buttonPlayerId)
{
	return (buttonPlayerId + 2) % Constants::nPlayers;
}

struct RoundAbstractions
{
	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> flopHistogramsMmap; 
	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> turnHistogramsMmap;
	std::unique_ptr<Histogram<AbstractionsContext::nOCHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nOCHSHistogramsBins>>> riverHistogramsMmap;

	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> flopCentroidsMmap;
	std::unique_ptr<Histogram<AbstractionsContext::nEHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nEHSHistogramsBins>>> turnCentroidsMmap;
	std::unique_ptr<Histogram<AbstractionsContext::nOCHSHistogramsBins>[], Memory::MmapDeleter<Histogram<AbstractionsContext::nOCHSHistogramsBins>>> riverCentroidsMmap;

	std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> flopHistograms;
	std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> turnHistograms;
	std::span<Histogram<AbstractionsContext::nOCHSHistogramsBins>> riverHistograms;

	std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> flopCentroids;
	std::span<Histogram<AbstractionsContext::nEHSHistogramsBins>> turnCentroids;
	std::span<Histogram<AbstractionsContext::nOCHSHistogramsBins>> riverCentroids;
};
RoundAbstractions roundAbstractions;

struct GameContext
{
	GameContext()
	{
		m_deck.shuffle();
	}

	FoldedPlayers m_foldedPlayersStatus{ 0 };
	BettingRound m_currentBettingRound{ BettingRound::Preflop };
	int m_roundActionIndex {0};
	int m_nBetInCurrentRound{ 0 };
	std::array<int, Constants::nPlayers> m_stacks{ 0 };
	std::array<int, Constants::nPlayers> m_moneyInPot{ 0 };
	Deck m_deck{ };
	int m_pot{ 0 };

	int m_currentButtonPlayerId{ 0 };
	int m_updatingPlayerId{ 0 };
	int m_currentPlayerTurn{ 0 };
	int m_lastBettingPlayer{ 0 };

	int m_currentBet{ 0 };
	
	GameHistory m_history {};

	// NOTE (keb): The BB has checked on the BB Preflop exception.
	bool m_isCheckAfterBBPreflopException {false};
	// NOTE (keb): When no bets have occured Preflop and it's the BB's turn.
	bool isBBPreflopException() const 
	{ 
		return m_lastBettingPlayer == getBBPlayerIdFromButton(m_currentButtonPlayerId) && 
			m_currentPlayerTurn == getBBPlayerIdFromButton(m_currentButtonPlayerId) &&
			m_nBetInCurrentRound == 1 && 
			m_currentBettingRound == BettingRound::Preflop; 
	}

	bool allinState()
	{
		int nPlayersAllin {0};
		for (int i = 0; i < Constants::nPlayers; ++i)
		{
			if (!m_foldedPlayersStatus.test(i) && m_stacks[i] <= 0)
			{
				++nPlayersAllin;
			}
		}
		int nPlayersRemaining = Constants::nPlayers - m_foldedPlayersStatus.nSet();
		return nPlayersAllin >= nPlayersRemaining-1;
	}

	void resetForNextPlayer(int updatingPlayerId)
	{
		m_history.reset();
		m_stacks.fill(500);
		m_moneyInPot.fill(0);
		m_foldedPlayersStatus.reset();
		m_currentBettingRound = BettingRound::Preflop;
		m_roundActionIndex = 0;
		m_nBetInCurrentRound = 0;
		m_pot = 0;
		m_currentButtonPlayerId = 0;
		m_updatingPlayerId = updatingPlayerId;
		m_currentPlayerTurn = 0;
		m_lastBettingPlayer = 0;
		m_currentBet = 0;
		m_isCheckAfterBBPreflopException = false;
	}

};

class Infoset
{
public:
	struct Key
	{
		// NOTE (keb): 60 bits for at most 5 actions per betting round * 4 betting rounds * 3 bits per action
		// TODO (keb): not flexible, can't add more players
		uint64_t history : 60;
		uint16_t clusterIndex : 8;
		uint8_t playerid : 2;			
		uint8_t stacks : 2*3;
		uint8_t pot : 3;
		
		bool operator==(const Key &other) const
		{ return (history == other.history 
			&& playerid == other.playerid
			&& clusterIndex == other.clusterIndex
			&& stacks == other.stacks
			&& pot == other.pot);
		}
	};

	struct KeyHasher
	{
		std::size_t operator()(const Key& k) const
		{
			using std::size_t;
			using std::hash;
			using std::string;

			std::size_t res = 17;
			res = res * 31 + hash<uint64_t>()( k.history);
			res = res * 31 + hash<uint16_t>()( k.clusterIndex);
			res = res * 31 + hash<uint8_t>()( k.playerid);
			res = res * 31 + hash<uint8_t>()( k.stacks);
			res = res * 31 + hash<uint8_t>()( k.pot);
			return res;
		}
	};

	std::array<int, Action::Count>& getRegrets() { return regrets; }
	void updateRegrets(const std::array<float, Action::Count>& values, Actions allowedActions, 
		float value)
	{
		for (int i = 0; i < Action::Count; ++i)
		{
			Action a = static_cast<Action>(i);
			if (!allowedActions.test(a))
			{
				continue;
			}
			regrets[a] += static_cast<int>(values[a] - value);
		}
		if (nTrainingIterations == 0) ++DebugGlobals::nTrained;
		++nTrainingIterations;
	}

	Actions getActions() { return allowedActions; }
	void setActions(const GameContext& gameContext) 
	{
		int pot = gameContext.m_pot;
		int stack = gameContext.m_stacks[gameContext.m_currentPlayerTurn];

		if (stack <= 0)
		{
			return;
		}
		if (gameContext.isBBPreflopException())
		{
			allowedActions.set(Check);
		}
		else
		{
			allowedActions.set(Call);
		}
		if (gameContext.m_nBetInCurrentRound > 0)
		{
			allowedActions.set(Fold);
		}
		if ((gameContext.m_currentBettingRound == BettingRound::Preflop && gameContext.m_nBetInCurrentRound >= Constants::maxNBetsPreflop) ||
			(gameContext.m_currentBettingRound != BettingRound::Preflop && gameContext.m_nBetInCurrentRound >= Constants::maxNBetsForRound))
		{
			return;
		}
		if (stack > 0)
		{
			allowedActions.set(BetHalf);
		}
		if (stack > 0.5*pot)
		{
			allowedActions.set(BetPot);
		}
		if (stack > pot)
		{
			allowedActions.set(BetOver);
		}
	}

	const std::array<float, Action::Count> calculateStrategy()
	{
		std::array<float, Action::Count> strategy{ 0.0 };
		int sum = 0;
		for (int a = 0; a < Action::Count; ++a)
		{
			Action action = static_cast<Action>(a);
			if (allowedActions.test(action))
			{
				sum += regrets[a] <= 0 ? 0 : regrets[a];
			}
		}

		int nAllowedActions = __builtin_popcount(allowedActions.flags);
		for (int a = 0; a < Action::Count; ++a)
		{
			Action action = static_cast<Action>(a);
			if (allowedActions.test(action))
			{
				strategy[a] = sum > 0.0 ? ((regrets[a] <= 0 ? 0 : regrets[a]) / static_cast<float>(sum)) : (1.0 / nAllowedActions);
			}
		}

		return strategy;
	}

	void discountRegrets(double discount)
	{
		for (int a = 0; a < Action::Count; ++a)
		{
			Action action = static_cast<Action>(a);
			if (allowedActions.test(action))
			{
				regrets[a] = static_cast<int>(regrets[a] * discount);
			}
		}
	}

	// const std::array<float, Action::Count>& getStrategy() { return strategy; }

private:
	// std::array<float, Action::Count> strategy{ 0.0 };
	std::array<int, Action::Count> regrets{ 0 };
	Actions allowedActions{0};
	uint8_t nTrainingIterations {0};
};

inline std::unordered_map<Infoset::Key, Infoset, Infoset::KeyHasher> infosets;

enum AbstractedSizeRatio : uint8_t
{
	Zero,
	Quarter,
	Half,
	Same,
	Double,
};

std::string toString(AbstractedSizeRatio ratio)
{
	if (ratio == Zero)
		return "Z";
	if (ratio == Quarter)
		return "Q";
	if (ratio == Half)
		return "H";
	if (ratio == Same)
		return "S";
	if (ratio == Double)
		return "D";
	return "?";
}

double toDouble(Action actionSize)
{
	switch (actionSize)
	{
	case BetHalf:
		return 0.5;
	case BetPot:
		return 1.0;
	case BetOver:
		return 3.0;
	default:
		return 0.0;
	}
}


// C.f. [PAPER] doi/abs/10.5555/2540128.2540148
double pseudoHarmonicMapping(double min, double max, double v)
{
	return ((max - v) * (1.0 + min)) / ((max - min) * (1.0 + v));
}

std::array<AbstractedSizeRatio, Constants::nPlayers> abstractStacks(int currentPlayerId, const std::array<int, Constants::nPlayers>& stacks)
{
	// A player's stack is abstracted to be a multiple of the current player's stack.
	std::array<AbstractedSizeRatio, Constants::nPlayers> abstraction;

	const int currentPlayerStack = stacks[currentPlayerId];
	for (int i = 0; i < Constants::nPlayers; ++i)
	{
		double ratio = static_cast<double>(currentPlayerStack) / stacks[i];
		if (ratio >= 2.0)
		{
			abstraction[i] = Double;
		}
		else if (ratio <= 0.5)
		{
			abstraction[i] = Half;
		}
		else
		{
			double probability = 1.0f;
			if (ratio > 0.5 && ratio <= 1.0)
			{
				probability = pseudoHarmonicMapping(0.5, 1.0, ratio);
				abstraction[i] = (Random::rand() <= probability) ? Half : Same;
			}
			else
			{
				probability = pseudoHarmonicMapping(1.0, 2.0, ratio);
				abstraction[i] = (Random::rand() <= probability) ? Same : Double;
			}
		}
	}
	return abstraction;
}

AbstractedSizeRatio abstractPot(int pot, int stack)
{
	// Half, Same, Double, X5
	double ratio = static_cast<double>(pot) / stack;
	if (ratio <= 0.5)
	{
		return Half;
	}
	if (ratio >= 2.0)
	{
		return Double;
	}
	
	double probability = 1.0f;
	if (ratio > 0.5 && ratio <= 1.0)
	{
		probability = pseudoHarmonicMapping(0.5, 1.0, ratio);
		return (Random::rand() <= probability) ? Half : Same;
	}
	else
	{
		probability = pseudoHarmonicMapping(1.0, 2.0, ratio);
		return (Random::rand() <= probability) ? Same : Double;
	}
}

// TODO: horrible code, templatize?
std::string getHandStr(int playerId, const Deck& deck, BettingRound round)
{
	std::string result;
	switch (round)
	{
	case BettingRound::Preflop:
	{
		const auto& hand = deck.getPreflopHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	case BettingRound::Flop:
	{
		const auto& hand = deck.getFlopHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	case BettingRound::Turn:
	{
		const auto& hand = deck.getTurnHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	case BettingRound::River:
	{
		const auto& hand = deck.getRiverHand(playerId);
		for (auto card : hand)
		{
			result += Constants::RANK_TO_CHAR[card.rank];
			result += Constants::SUIT_TO_CHAR[card.suit];
		}
		result += '|';
		return result;
	}
	default:
	{
		return "?";
	}
	}
	return "?";
}

// TODO: horrible code, templatize?
uint16_t getAbstractionIndex(int playerId, const Deck& deck, BettingRound round)
{
	uint16_t result;
	switch (round)
	{
	case BettingRound::Preflop:
	{
		const auto& hand = deck.getPreflopHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::Preflop]];
		std::copy(hand.begin(), hand.end(), cards);
		result = hand_index_last(&indexers[(uint8_t)round], cards);
		return result;
	}
	case BettingRound::Flop:
	{
		using FlopHistogram = Histogram<AbstractionsContext::nEHSHistogramsBins>;
		const auto& hand = deck.getFlopHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::Flop]];
		std::copy(hand.begin(), hand.end(), cards);
		uint64_t handIndex = hand_index_last(&indexers[(uint8_t)round], cards);
		const auto& histogram = roundAbstractions.flopHistograms[handIndex];
		result = KMeans::findNearestCentroid<FlopHistogram, KMeans::L2Distance<FlopHistogram>>(roundAbstractions.flopCentroids, histogram).first;
		return result;
	}
	case BettingRound::Turn:
	{
		using TurnHistogram = Histogram<AbstractionsContext::nEHSHistogramsBins>;
		const auto& hand = deck.getTurnHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::Turn]];
		std::copy(hand.begin(), hand.end(), cards);
		uint64_t handIndex = hand_index_last(&indexers[(uint8_t)round], cards);
		const auto& histogram = roundAbstractions.turnHistograms[handIndex];
		result = KMeans::findNearestCentroid<TurnHistogram, KMeans::L2Distance<TurnHistogram>>(roundAbstractions.turnCentroids, histogram).first;
		return result;
	}
	case BettingRound::River:
	{
		using RiverHistogram = Histogram<AbstractionsContext::nOCHSHistogramsBins>;
		const auto& hand = deck.getRiverHand(playerId);
		uint8_t cards[Constants::nCardsRoundAccumulator[(uint8_t)BettingRound::River]];
		std::copy(hand.begin(), hand.end(), cards);
		uint64_t handIndex = hand_index_last(&indexers[(uint8_t)round], cards);
		const auto& histogram = roundAbstractions.riverHistograms[handIndex];
		result = KMeans::findNearestCentroid<RiverHistogram, KMeans::L2Distance<RiverHistogram>>(roundAbstractions.riverCentroids, histogram).first;
		return result;
	}
	default:
	{
		return std::numeric_limits<uint16_t>::max();
	}
	}
	return std::numeric_limits<uint16_t>::max();
}

uint64_t serializeHistory(const GameHistory& history)
{
	uint64_t result {0};
	for (int i = 0; i < (uint8_t)BettingRound::RoundCount; ++i)
	{
		for (int j = 0; j < Constants::maxNActionsPerRound; ++j)
		{
			Action a = history[i][j];
			result |= (uint64_t)a << ((i*Constants::maxNActionsPerRound*3)+(j*3));
		}
	}
	return result;
}
	
// NOTE: a game history is encoded like so:
//       actions/actions/actions/actions, where the '/' seperates the betting rounds.
//       An information set key is an encoding of the playerId, the information abstraction index,
//       the game history, the players' stacks relative to the current state player, 
//       the pot size relative to the current state player. 
//		 All are divided by '|', like so: 
//       playerId|clusterIndex|actions/actions/actions|ActivePlayer1Stack/ActivePlayer2Stack|RelativePot
Infoset::Key makeInfosetKey(const GameHistory& history,
	int playerId,
	const Deck& deck,
	BettingRound bettingRound,
	std::array<AbstractedSizeRatio, Constants::nPlayers> stacks, 
	AbstractedSizeRatio pot,
	FoldedPlayers foldedPlayers)
{
	// TODO (keb): review this
	Infoset::Key key;
	uint16_t clusterIndex = getAbstractionIndex(playerId, deck, bettingRound);
	assert(clusterIndex < AbstractionsContext::maxNBuckets && "Invalid Information Abstraction Cluster Index");
	key.clusterIndex = clusterIndex;
	key.playerid = playerId;
	key.history = serializeHistory(history);

	// NOTE (keb): Uncomment if you want to consider the relative stack and pot sizes.
	// key.stacks = 0;
	// for (int i = 0; i < Constants::nPlayers; ++i)
	// {
	// 	bool isFolded = foldedPlayers.test(i);
	// 	// NOTE (keb): shift bits by 3 because we reserve 3 bits to represent a stack.
	// 	key.stacks |= (isFolded ? ((uint8_t)Zero << (i*3)) : (uint8_t(stacks[i]) << (i*3)));
	// }
	// key.pot = (uint8_t)pot;
	
	return key;
}

Action sampleAction(double rand, const std::array<float, Action::Count>& strategy, Actions allowedActions)
{
	float randAcc = 0.0;
	for (int i = 0; i < strategy.size(); ++i)
	{
		if (!allowedActions.test(static_cast<Action>(i)))
		{
			continue;
		}
		randAcc += strategy[i];
		if (rand <= randAcc)
		{
			return static_cast<Action>(i);
		}
	}
	return Action::Count;
}

void call(GameContext& gameContext)
{
	auto& stack = gameContext.m_stacks[gameContext.m_currentPlayerTurn];
	int amount = gameContext.m_currentBet > stack ? stack : gameContext.m_currentBet;
	stack -= amount;
	gameContext.m_pot += amount;
	gameContext.m_moneyInPot[gameContext.m_currentPlayerTurn] += amount;
}

void undoCall(GameContext& gameContext, int previousStack, int previousBet, int previousPot)
{
	auto& stack = gameContext.m_stacks[gameContext.m_currentPlayerTurn];
	int amount = previousBet > previousStack ? previousStack : previousBet;
	stack += amount;
	gameContext.m_pot -= amount;
	gameContext.m_moneyInPot[gameContext.m_currentPlayerTurn] -= amount;
}

void check(GameContext& gameContext)
{
	gameContext.m_isCheckAfterBBPreflopException = true;
}
void undoCheck(GameContext& gameContext)
{
	gameContext.m_isCheckAfterBBPreflopException = false;
}

void bet(int playerId, int amount, GameContext& gameContext)
{
	gameContext.m_moneyInPot[playerId] += amount;
	gameContext.m_stacks[playerId] -= amount;
	gameContext.m_pot += amount;
	gameContext.m_currentBet = amount;
	++gameContext.m_nBetInCurrentRound;
	gameContext.m_lastBettingPlayer = playerId;
}

void undoBet(int playerId, int amount, GameContext& gameContext, int previousBet, int previousPot, int previousLastBettingPlayer)
{
	gameContext.m_moneyInPot[playerId] -= amount;
	gameContext.m_stacks[playerId] += amount;
	gameContext.m_pot -= amount;
	gameContext.m_currentBet = previousBet;
	--gameContext.m_nBetInCurrentRound;
	gameContext.m_lastBettingPlayer = previousLastBettingPlayer;
}

void betRelativePot(GameContext& gameContext, double ratio)
{
	int pot = gameContext.m_pot;
	int stack = gameContext.m_stacks[gameContext.m_currentPlayerTurn];
	int amount = stack >= ratio * pot ? static_cast<int>(ratio * pot) : stack;
	bet(gameContext.m_currentPlayerTurn, amount, gameContext);
}

void undoBetRelativePot(GameContext& gameContext, double ratio, int previousStack, int previousBet, int previousPot, int previousLastBettingPlayer)
{
	int amount = previousStack >= ratio * previousPot ? static_cast<int>(ratio * previousPot) : previousStack;
	undoBet(gameContext.m_currentPlayerTurn, amount, gameContext, previousBet, previousPot, previousLastBettingPlayer);
}

void applyAction(GameContext& gameContext, Action a)
{
	switch (a)
	{
	case Fold:
	{
		gameContext.m_foldedPlayersStatus.set(gameContext.m_currentPlayerTurn);
		break;
	}
	case Call:
	{
		call(gameContext);
		break;
	}
	case Check:
	{
		check(gameContext);
		break;
	}
	case BetHalf:
	case BetPot:
	case BetOver:
	{
		betRelativePot(gameContext, toDouble(a));
		break;
	}
	default:
	{
		std::cerr << "Applying Invalid Action" << std::endl;
	}
	}
}

void unApplyAction(GameContext& gameContext, Action a, int previousStack, int previousBet, int previousPot, int previousLastBettingPlayer)
{
	switch (a)
	{
	case Fold:
	{
		gameContext.m_foldedPlayersStatus.unset(gameContext.m_currentPlayerTurn);
		break;
	}
	case Call:
	{
		undoCall(gameContext, previousStack, previousBet, previousPot);
		break;
	}
	case Check:
	{
		undoCheck(gameContext);
		break;
	}
	case BetHalf:
	case BetPot:
	case BetOver:
	{
		undoBetRelativePot(gameContext, toDouble(a), previousStack, previousBet, previousPot, previousLastBettingPlayer);
		break;
	}
	default:
	{
		std::cerr << "Unapplying Invalid Action" << std::endl;
	}
	}
}

struct PreviousBettingRoundSnapshot
{
	bool hasSnapped {false};
	int playerId;
	int lastBettingPlayerId;
	int currentBet;
	int nBet;
	BettingRound currentBettingRound;
	int roundActionIndex;
};

void snapBettingRound(const GameContext& gameContext, PreviousBettingRoundSnapshot& snapshot)
{
	snapshot.hasSnapped = true;
	snapshot.playerId = gameContext.m_currentPlayerTurn;
	snapshot.lastBettingPlayerId = gameContext.m_lastBettingPlayer;
	snapshot.currentBet = gameContext.m_currentBet;
	snapshot.nBet = gameContext.m_nBetInCurrentRound;
	snapshot.currentBettingRound = gameContext.m_currentBettingRound;
	snapshot.roundActionIndex = gameContext.m_roundActionIndex;
}

void undoBettingRound(const PreviousBettingRoundSnapshot& snapshot, GameContext& gameContext)
{
	if (!snapshot.hasSnapped)
	{
		return;
	}
	gameContext.m_currentPlayerTurn = snapshot.playerId;
	gameContext.m_lastBettingPlayer = snapshot.lastBettingPlayerId;
	gameContext.m_currentBet = snapshot.currentBet;
	gameContext.m_nBetInCurrentRound = snapshot.nBet;
	gameContext.m_currentBettingRound = snapshot.currentBettingRound;
	gameContext.m_roundActionIndex = snapshot.roundActionIndex;
}

std::optional<float> isTerminalState(GameContext& gameContext)
{
	if (gameContext.m_foldedPlayersStatus.test(gameContext.m_updatingPlayerId))
	{
		return -gameContext.m_moneyInPot[gameContext.m_updatingPlayerId];
	}
	
	if (!gameContext.m_foldedPlayersStatus.test(gameContext.m_updatingPlayerId) &&
		gameContext.m_foldedPlayersStatus.one())
	{
		return gameContext.m_pot;
	}

	// River Showdown
	if ((gameContext.m_currentPlayerTurn == gameContext.m_lastBettingPlayer && !(gameContext.isBBPreflopException())) && 
		gameContext.m_currentBettingRound == BettingRound::River)
	{
		int bestHandSeen = INT_MAX;
		std::unordered_set<int> tyingPlayerIds{};
		for (int i = 0; i < Constants::nPlayers; ++i)
		{
			if (gameContext.m_foldedPlayersStatus.test(i))
			{
				continue;
			}
			auto hand = gameContext.m_deck.getRiverHand(i);
			if (int handStrength = std::apply(evaluate_7cards, hand); handStrength <= bestHandSeen) // smaller index is stronger
			{
				if (handStrength == bestHandSeen)
				{
					tyingPlayerIds.insert(i);
					continue;
				}
				tyingPlayerIds.clear();
				tyingPlayerIds.insert(i);
				bestHandSeen = handStrength;
			}
		}
		if (tyingPlayerIds.find(gameContext.m_updatingPlayerId) != tyingPlayerIds.end())
		{
			return static_cast<int>(gameContext.m_pot / tyingPlayerIds.size()); // Integer Division, inaccuracy negligible 
		}
		else
		{
			return -gameContext.m_moneyInPot[gameContext.m_updatingPlayerId];
		}
	}
	return std::nullopt; 
}

float traverseMccfrWithNewAction(Action a, GameContext& gameContext, 
	float& actionValue, float strategy, bool isPruning)
{
	float result = 0.0f;
	gameContext.m_history[(uint8_t)gameContext.m_currentBettingRound][gameContext.m_roundActionIndex] = a;
	++gameContext.m_roundActionIndex;
	int previousStack = gameContext.m_stacks[gameContext.m_currentPlayerTurn];
	int previousBet = gameContext.m_currentBet;
	int previousPot = gameContext.m_pot;
	int previousLastBettingPlayer = gameContext.m_lastBettingPlayer;
	applyAction(gameContext, a);
	gameContext.m_currentPlayerTurn = getNextPlayer(gameContext.m_currentPlayerTurn);
	actionValue = traverseMccfr(gameContext, isPruning);
	gameContext.m_currentPlayerTurn = getPreviousPlayer(gameContext.m_currentPlayerTurn);
	unApplyAction(gameContext, a, previousStack, previousBet, previousPot, previousLastBettingPlayer);
	result = strategy * actionValue;
	--gameContext.m_roundActionIndex;
	gameContext.m_history[(uint8_t)gameContext.m_currentBettingRound][gameContext.m_roundActionIndex] = Action::Count;
	return result;
}

float traverseMccfr(GameContext& gameContext, const bool isPruning = false)
{
	if (auto payoff = isTerminalState(gameContext); payoff)
	{
		return payoff.value();
	}
	
	// Betting Round Complete
	PreviousBettingRoundSnapshot previousRoundSnapshot;
	if ((gameContext.m_currentPlayerTurn == gameContext.m_lastBettingPlayer && !(gameContext.isBBPreflopException())) ||
		gameContext.m_isCheckAfterBBPreflopException ||
		gameContext.allinState())
	{
		gameContext.m_isCheckAfterBBPreflopException = false;
		
		snapBettingRound(gameContext, previousRoundSnapshot);
		// Move to the next betting round
		gameContext.m_currentBettingRound = (BettingRound) ((uint8_t)gameContext.m_currentBettingRound + 1);
		gameContext.m_currentPlayerTurn = getSBPlayerIdFromButton(gameContext.m_currentButtonPlayerId); // The SB starts the new round.
		gameContext.m_lastBettingPlayer = gameContext.m_currentPlayerTurn;
		gameContext.m_currentBet = 0;
		gameContext.m_nBetInCurrentRound = 0;
		gameContext.m_roundActionIndex = 0;
	}

	float value = 0.0;
	if (gameContext.allinState())
	{
		value = traverseMccfr(gameContext, isPruning);
	}
	else if (gameContext.m_foldedPlayersStatus.test(gameContext.m_currentPlayerTurn))
	{
		gameContext.m_currentPlayerTurn = getNextPlayer(gameContext.m_currentPlayerTurn);
		value = traverseMccfr(gameContext, isPruning);
		gameContext.m_currentPlayerTurn = getPreviousPlayer(gameContext.m_currentPlayerTurn);
	}
	else
	{
		auto abstractedStacks = abstractStacks(gameContext.m_currentPlayerTurn, gameContext.m_stacks);
		auto abstractedPot = abstractPot(gameContext.m_pot, gameContext.m_stacks[gameContext.m_currentPlayerTurn]);
		Infoset::Key infosetKey = makeInfosetKey(gameContext.m_history, gameContext.m_currentPlayerTurn, gameContext.m_deck, gameContext.m_currentBettingRound, abstractedStacks, abstractedPot, gameContext.m_foldedPlayersStatus);
		auto [infosetItr, isNewState] = infosets.try_emplace(infosetKey, Infoset());
		if (isNewState)
		{
			infosetItr->second.setActions(gameContext);
			infosetItr->second.calculateStrategy();
		}

		Infoset& infoset = infosetItr->second;
		const auto strategy = infoset.calculateStrategy();
		if (gameContext.m_currentPlayerTurn == gameContext.m_updatingPlayerId)
		{
			++DebugGlobals::visited;
			Actions allowedActions = infoset.getActions();
			std::array<float, Action::Count> actionValues{ 0 };
			for (int i = 0; i < Action::Count; ++i)
			{
				Action a = static_cast<Action>(i);
				if (!allowedActions.test(a))
				{
					continue;
				}
				if (isPruning && infoset.getRegrets()[a] < Constants::pruningRegretThreshold)
				{
					allowedActions.unset(a);
					continue;
				}
				value += traverseMccfrWithNewAction(a, gameContext, actionValues[i], strategy[i], isPruning);
			}

			infoset.updateRegrets(actionValues, allowedActions, value);	
			// TODO (keb): Redundant?
			infoset.calculateStrategy();
		}
		else
		{
			Random::rand();
			Action a = sampleAction(Random::rand(), strategy, infoset.getActions());
			traverseMccfrWithNewAction(a, gameContext, value, 1, isPruning);
		}
	}
	
	undoBettingRound(previousRoundSnapshot, gameContext);
	return value;
}

void initAbstractions()
{
	using namespace std::chrono_literals;
	using namespace PokerTypes::AbstractionsContext;
	
	const uint8_t preflop[] = {2};
	const uint8_t flop[] = {2,3};
	const uint8_t turn[] = {2,3,1};
	const uint8_t river[] = {2,3,1,1};
	assert(hand_indexer_init(1, preflop, &indexers[0]));
	assert(hand_indexer_init(2, flop, &indexers[1]));
	assert(hand_indexer_init(3, turn, &indexers[2]));
	assert(hand_indexer_init(4, river, &indexers[3]));
	
	const auto nFlopSize = hand_indexer_size(&indexers[(uint8_t)BettingRound::River], (uint8_t)BettingRound::Flop);
	const auto nTurnSize = hand_indexer_size(&indexers[(uint8_t)BettingRound::River], (uint8_t)BettingRound::Turn);
	const auto nRiverSize = hand_indexer_size(&indexers[(uint8_t)BettingRound::River], (uint8_t)BettingRound::River);
	
	roundAbstractions.flopHistogramsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + flopHistogramsFilename).c_str(), nFlopSize);
	roundAbstractions.turnHistogramsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + turnHistogramsFilename).c_str(), nTurnSize);
	roundAbstractions.riverHistogramsMmap = Memory::getMmap<Histogram<nOCHSHistogramsBins>>((std::string("../EHS/") + riverOCHSFilename).c_str(), nRiverSize);
	
	roundAbstractions.flopCentroidsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + flopCentroidsFilename).c_str(), nFlopBuckets);
	roundAbstractions.turnCentroidsMmap = Memory::getMmap<Histogram<nEHSHistogramsBins>>((std::string("../EHS/") + turnCentroidsFilename).c_str(), nTurnBuckets);
	roundAbstractions.riverCentroidsMmap = Memory::getMmap<Histogram<nOCHSHistogramsBins>>((std::string("../EHS/") + riverCentroidsFilename).c_str(), nRiverBuckets);
	
	roundAbstractions.flopHistograms = {roundAbstractions.flopHistogramsMmap.get(), roundAbstractions.flopHistogramsMmap.get() + nFlopSize};
	roundAbstractions.turnHistograms = {roundAbstractions.turnHistogramsMmap.get(), roundAbstractions.turnHistogramsMmap.get() + nTurnSize};
	roundAbstractions.riverHistograms = {roundAbstractions.riverHistogramsMmap.get(), roundAbstractions.riverHistogramsMmap.get() + nRiverSize};
	
	roundAbstractions.flopCentroids = {roundAbstractions.flopCentroidsMmap.get(), roundAbstractions.flopCentroidsMmap.get() + nFlopBuckets};
	roundAbstractions.turnCentroids = {roundAbstractions.turnCentroidsMmap.get(), roundAbstractions.turnCentroidsMmap.get() + nTurnBuckets};
	roundAbstractions.riverCentroids = {roundAbstractions.riverCentroidsMmap.get(), roundAbstractions.riverCentroidsMmap.get() + nRiverBuckets};
}
			
void placeBlinds(GameContext& gameContext)
{
	bet(getSBPlayerIdFromButton(gameContext.m_currentButtonPlayerId), Constants::sb, gameContext);
	bet(getBBPlayerIdFromButton(gameContext.m_currentButtonPlayerId), Constants::bb, gameContext);
	gameContext.m_nBetInCurrentRound = 1;
}

void updateEachPlayer(GameContext& gameContext, bool isPruningIteration)
{
	for (int playerId = 0; playerId < Constants::nPlayers; ++playerId)
	{
		gameContext.resetForNextPlayer(playerId);
		gameContext.m_updatingPlayerId = playerId;
		gameContext.m_currentPlayerTurn = getNextPlayer(getBBPlayerIdFromButton(gameContext.m_currentButtonPlayerId));
		placeBlinds(gameContext);
		traverseMccfr(gameContext, isPruningIteration);
	}
}

void snapStrategies(int t)
{
	// TODO (keb): Take a snapshot of the current strategy to make an average of the snapshots later.
	// for (player in players)
	// if (t % Constants::strategyInterval == 0)
	// {
	// 	updateStrategy(_, playerIdx);
	// }
}

void discountRegrets(int timeElapsed)
{
	if (timeElapsed < Constants::LCFRThresholdMinutes && 
		(timeElapsed % Constants::discountIntervalMinutes == 0))
	{
		double discount = (timeElapsed / static_cast<double>(Constants::discountIntervalMinutes)) / ((timeElapsed / static_cast<double>(Constants::discountIntervalMinutes)) + 1);
		for (auto& [key, infoset] : infosets)
		{
			infoset.discountRegrets(discount);
		}
	}
}

void mccfrP(int nIterations)
{
	using namespace std::chrono_literals;
	
	initAbstractions();
		
	auto startTime = std::chrono::steady_clock::now();

	for (int t = 0; t < nIterations; ++t)
	{
		GameContext gameContext{};

		snapStrategies(t);

		bool isPruningIteration = false;
		const auto timeElapsed = Time::getTimeElapsed(startTime, 1min);
		if (timeElapsed > Constants::pruningThresholdMinutes)
		{
			isPruningIteration = Random::rand() < 0.95;
		}
		updateEachPlayer(gameContext, isPruningIteration);

		discountRegrets(timeElapsed);
	}
}

};

int main()
{
	MCCFR::mccfrP(10000000);
	return 0;
}