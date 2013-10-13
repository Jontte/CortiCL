#ifndef SENSOR_H_INCLUDED
#define SENSOR_H_INCLUDED

#include <vector>

// This class defines a dynamic encoding from different data sources to SDRs and back

class Sensor
{
	static const int SAMPLE_WINDOW_SIZE = 1000; // how many samples to consider when approximating
private:

	int m_totalSize;
	int m_windowSize;
	std::vector<double> m_samples;
	std::vector<double> m_histogram;

	void compute_histogram();

public:

	Sensor(int totalSize, int windowSize) : m_totalSize(totalSize), m_windowSize(windowSize){}
	std::vector<signed char> encode(double value, bool learning = true);
	double decode(const std::vector<double>& sdr);

	const std::vector<double>& getHistogram() const
	{
		return m_histogram;
	}
};

#endif
