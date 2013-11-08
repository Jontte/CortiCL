#ifndef CLREGION_H_INCLUDED
#define CLREGION_H_INCLUDED

#include <iostream>
#include <string>
#include <memory>
#include <map>

#include "clcontext.h"
#include "clspatial.h"
#include "cltemporal.h"
#include "cltopology.h"
#include "clargs.h"

std::string getCLError(cl_int err);

struct CLStats
{
	// Spatial pooler
	double averageBoost;
	double averageDutyCycle;

	// Temporal pooler
	int predictiveState; // number of cells in predictive state
	int activeState; // number of cells in active state
	int learningState; // number of cells in learning state
	double averageSegmentDutyCycle;
};

class CLRegion
{
private:
	CLContext& m_context;

	CLSpatialPooler m_spatialPooler;
	CLTemporalPooler m_temporalPooler;

public:

	CLRegion(CLContext& context, const CLTopology& topo, const CLArgs& args);

	CLRegion(const CLRegion&) = delete;
	CLRegion(CLRegion&&) = default;

	// Primary input function
	void write(std::vector<cl_char> & activations, std::vector<cl_char>& results, bool temporal = true);

	// Noisy backwards convolution: Find out what kind of bit pattern would cause the given column activation
	void backwards(const std::vector<cl_char>& columnActivation, std::vector<double>& result);

	// Read statistics from network. This can be very expensive as the full network has to be downloaded from the computing device.
	CLStats getStats();
};

#endif
