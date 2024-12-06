#pragma once

#include <random>

namespace Random
{
	inline std::mt19937 generate()
	{
		std::random_device rd{};
		std::seed_seq ss{ rd(), rd(), rd() };
		return std::mt19937{ ss };
	}

	inline std::mt19937 mt{ generate() };

	template <typename T>
	T get(T min, T max)
	{
		return std::uniform_int_distribution<T>{min, max}(mt);
	}

	double rand()
	{
		return std::uniform_real_distribution{ 0.0, 1.0 } (mt);
	}

	bool flipCoin()
	{
		return std::uniform_int_distribution{ 0, 1 } (mt);
	}
};
