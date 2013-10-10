#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include <iostream>
#include <chrono>
#include <string>

class Timer
{
	typedef std::chrono::steady_clock clock_type ;
	typedef std::chrono::milliseconds milliseconds;
public:
	explicit Timer(bool run = true)
	{
		if (run)
			Reset();
	}
	void Reset()
	{
		_start = clock_type::now();
	}
	double Elapsed() const
	{
		return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1>>>(clock_type::now() - _start).count();
	}
	template <typename T, typename Traits>
	friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const Timer& timer)
	{
		return out << timer.Elapsed();
	}
private:
	clock_type::time_point _start;
};

class LabeledTimer
{
private:
	Timer m_timer;
	std::string m_label;
public:
	LabeledTimer(const std::string& label)
		: m_timer(true)
		, m_label(label){}
	~LabeledTimer()
	{
		std::cout << m_label << " done: " << m_timer << std::endl;
	}
};

// Shell color printing
enum class Color
{
	NONE = 0,
	BLACK, RED, GREEN,
	YELLOW, BLUE, MAGENTA,
	CYAN, WHITE
};

std::string set_color(Color foreground = Color::NONE, Color background = Color::NONE);


#endif