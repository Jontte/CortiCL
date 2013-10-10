#ifndef CLTEMPORAL_H_INCLUDED
#define CLTEMPORAL_H_INCLUDED

#include <string>
#include <CL/cl.hpp>
#include "cltopology.h"
#include "clargs.h"

std::string getCLError(cl_int err);

struct CLStats;
class CLTemporalPooler
{
private:

	struct CLSynapse
	{
		cl_float permanence;
		cl_float permanenceQueued;
		cl_int targetColumn;
		cl_uchar targetCell;
		cl_uchar targetCellState;

		// Bits in Cellstate:
		// 0 = active
		// 1 = predictive
		// 2 = learning
		// 3
		// 4 = was active
		// 5 = was predictive
		// 6 = was learning
		// 7 =
	};
	struct CLSegment
	{
		cl_uchar synapseCount;

		// Activity of the segment
		// 0 = activeState, 1 = learnState
		// 0 = now, 1 = previous timestep
		cl_uchar activity[2][2];

		// Activity that includes synapses with permanence below CONNECTED_PERMANENCE but above MIN_PERMANENCE
		// 0 = activeState, 1 = learnState
		// 0 = now, 1 = previous timestep
		cl_uchar fullActivity[2][2];

		cl_bool sequenceSegment;
		cl_bool sequenceSegmentQueued;
		cl_bool hasQueuedChanges;
	};
	struct CLCell
	{
		cl_uchar segmentCount;
		cl_uchar newSegmentCount; // segments waiting to be added

		// See state definitions above
		cl_uchar state;
	};

	cl::Device& m_device;
	cl::Context& m_context;
	cl::CommandQueue& m_commandQueue;

	const CLTopology m_topology;
	const CLArgs m_args;

	cl::KernelFunctor m_timeStepKernel;
	cl::KernelFunctor m_computeActiveStateKernel;
	cl::KernelFunctor m_computePredictiveState;
	cl::KernelFunctor m_updateSynapsesKernel;

	std::vector<CLCell> m_cellData;
	std::vector<CLSegment> m_segmentData;
	std::vector<CLSynapse> m_synapseData;
	cl::Buffer m_cellDataBuffer;
	cl::Buffer m_segmentDataBuffer;
	cl::Buffer m_synapseDataBuffer;
	cl::Buffer m_inputDataBuffer;

	void pushBuffers(bool cells = true, bool segments = true, bool synapses = true);
	void pullBuffers(bool cells = true, bool segments = true, bool synapses = true);

public:

	CLTemporalPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, const CLTopology& topo, const CLArgs& args);
	void write(const std::vector< cl_char >& activations_in, std::vector<cl_char>& results_out);
	void getStats(CLStats& stats);
};

#endif
