#ifndef CLREGION_H_INCLUDED
#define CLREGION_H_INCLUDED

#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <CL/cl.hpp>

std::string getCLError(cl_int err);

#include "clspatial.h"
#include "cltemporal.h"
#include "cltopology.h"
#include "clargs.h"

struct CLStats
{
	// Spatial pooler
	double averageBoost;

	// Temporal pooler
	int totalSegments;
	int maxSegments;
	int totalSynapses;
	int maxSynapses;
	double averageDutyCycle;
};

class CLRegion
{
private:
	cl::CommandQueue m_commandQueue;

	CLSpatialPooler m_spatialPooler;
	CLTemporalPooler m_temporalPooler;

public:

	CLRegion(cl::Device& device, cl::Context& context, const CLTopology& topo, const CLArgs& args);

	CLRegion(const CLRegion&) = delete;
	CLRegion(CLRegion&&) = default;

	// Primary input function
	void write(std::vector< cl_char >& activations, std::vector< cl_char >& results, bool learning = true, bool temporal = true);

	// Noisy backwards convolution: Find out what kind of bit pattern would cause the given column activation
	void backwards(const std::vector<cl_char>& columnActivation, std::vector<double>& result);

	CLStats getStats();
};

#endif
