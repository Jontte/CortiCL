#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <random>
#include <algorithm>
#include <sstream>

#include "clregion.h"

constexpr static const char* TEMPORAL_SRC =
#include "temporal.cl.h"
;

CLTemporalPooler::CLTemporalPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, const CLTopology& topo, const CLArgs& args)
	: m_device(device)
	, m_context(context)
	, m_commandQueue(queue)
	, m_topology(topo)
	, m_args(args)
	, m_cellDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(CLCell) * args.ColumnCellCount)
	, m_segmentDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(CLSegment) * args.ColumnCellCount * args.CellSegmentCount)
	, m_synapseDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(CLSynapse) * args.ColumnCellCount * args.CellSegmentCount * args.SegmentSynapseCount)
	, m_inputDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(cl_char))
{
	std::cerr << "CLTemporalPooler: Initializing" << std::endl;

	// Install kernel programs
	std::string definitions = args.serialize() + topo.serialize();

	cl::Program::Sources sources;
	sources.push_back({definitions.c_str(), definitions.length()});
	sources.push_back({TEMPORAL_SRC, strlen(TEMPORAL_SRC)});

	cl::Program program(context, sources);
	try
	{
		program.build({device});
	}
	catch(const cl::Error& err)
	{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		throw;
	}

	m_timeStepKernel = cl::KernelFunctor(cl::Kernel(program, "timeStep"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_computeActiveStateKernel = cl::KernelFunctor(cl::Kernel(program, "computeActiveState"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_computePredictiveState = cl::KernelFunctor(cl::Kernel(program, "computePredictiveState"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_updateSynapsesKernel = cl::KernelFunctor(cl::Kernel(program, "updateSynapses"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);

	// Initialize all columns
	m_cellData.resize(m_topology.getColumns() * args.ColumnCellCount);
	m_segmentData.resize(m_topology.getColumns() * args.ColumnCellCount * args.CellSegmentCount);
	m_synapseData.resize(m_topology.getColumns() * args.ColumnCellCount * args.CellSegmentCount * args.SegmentSynapseCount);

	// Initialize region
	cl::KernelFunctor initRegion =
		cl::KernelFunctor(cl::Kernel(program, "initRegion"), m_commandQueue,
		cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);

	cl_uint2 randomState;
	randomState.s[0] = rand();
	randomState.s[1] = rand();
	initRegion(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer, randomState);
	std::cerr << "CLTemporalPooler: Kernels loaded" << std::endl;
}
void CLTemporalPooler::pullBuffers(bool cells, bool segments, bool synapses)
{
	if (cells)
		m_commandQueue.enqueueReadBuffer(m_cellDataBuffer, CL_FALSE, 0, sizeof(CLCell) * m_cellData.size(), &m_cellData[0]);
	if (segments)
		m_commandQueue.enqueueReadBuffer(m_segmentDataBuffer, CL_FALSE, 0, sizeof(CLSegment) * m_segmentData.size(), &m_segmentData[0]);
	if (synapses)
		m_commandQueue.enqueueReadBuffer(m_synapseDataBuffer, CL_FALSE, 0, sizeof(CLSynapse) * m_synapseData.size(), &m_synapseData[0]);
	m_commandQueue.finish();
}
void CLTemporalPooler::pushBuffers(bool cells, bool segments, bool synapses)
{
	if (cells)
		m_commandQueue.enqueueWriteBuffer(m_cellDataBuffer, CL_FALSE, 0, sizeof(CLCell) * m_cellData.size(), &m_cellData[0]);
	if (segments)
		m_commandQueue.enqueueWriteBuffer(m_segmentDataBuffer, CL_FALSE, 0, sizeof(CLSegment) * m_segmentData.size(), &m_segmentData[0]);
	if (synapses)
		m_commandQueue.enqueueWriteBuffer(m_synapseDataBuffer, CL_FALSE, 0, sizeof(CLSynapse) * m_synapseData.size() , &m_synapseData[0]);
	m_commandQueue.finish();
}

void CLTemporalPooler::write(const std::vector< cl_char >& activations_in, std::vector< cl_char >& results_out)
{
	if (activations_in.size() != std::size_t(m_topology.getColumns()))
	{
		throw std::runtime_error("Invalid vector length!");
	}

	// Send input column activations to device
	m_commandQueue.enqueueWriteBuffer(m_inputDataBuffer, CL_TRUE, 0, m_topology.getColumns() * sizeof(cl_char), &activations_in[0]);

	// provide GPU some poor man's randomness
	cl_uint2 randomSeed;
	randomSeed.s[0] = rand();
	randomSeed.s[1] = rand();

	// Phase 0: Step forwards in time
	m_timeStepKernel(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer);

	// Phase 1: Compute active state for each cell
	m_computeActiveStateKernel(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer, m_inputDataBuffer, randomSeed);

	// Phase 2: Compute predictive state for each cell
	m_computePredictiveState(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer, m_inputDataBuffer, randomSeed);

	// Phase 3: Update permanences
	m_updateSynapsesKernel(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer, m_inputDataBuffer);

	// Obtain result (list of column activity) from compute device and save to results_out
	results_out.resize(m_topology.getColumns());
	m_commandQueue.enqueueReadBuffer(m_inputDataBuffer, CL_TRUE, 0, sizeof(cl_char) * m_topology.getColumns(), &results_out[0]);
}

void CLTemporalPooler::getStats(CLStats& stats)
{
	pullBuffers();

	/*stats.maxSegments = m_topology.getColumns() * m_args.ColumnCellCount * m_args.CellMaxSegments;
	stats.maxSynapses = m_topology.getColumns() * m_args.ColumnCellCount * m_args.CellMaxSegments * m_args.SegmentMaxSynapses;
	stats.totalSegments = 0;
	stats.totalSynapses = 0;

	for (int i = 0; i < int(m_cellData.size()); ++i)
	{
		CLCell& cell = m_cellData[i];
		stats.totalSegments += cell.segmentCount;

		for (int a = 0; a < cell.segmentCount; ++a)
		{
			int offset = i * m_args.CellMaxSegments;
			CLSegment& seg = m_segmentData[offset+a];
			stats.totalSynapses += seg.synapseCount;
		}
	}*/
}
