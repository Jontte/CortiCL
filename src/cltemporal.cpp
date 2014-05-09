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

CLTemporalPooler::CLTemporalPooler(CLContext& context, const CLTopology& topo, const CLArgs& args)
	: m_context(context)
	, m_topology(topo)
	, m_args(args)
	, m_cellData(context, m_topology.getColumns() * args.ColumnCellCount)
	, m_segmentData(context, m_topology.getColumns() * args.ColumnCellCount * args.CellSegmentCount)
	, m_synapseData(context, m_topology.getColumns() * args.ColumnCellCount * args.CellSegmentCount * args.SegmentSynapseCount)
	, m_inputData(context, m_topology.getColumns())
{
	std::cerr << "CLTemporalPooler: Initializing" << std::endl;

	// Install kernel programs
	std::string definitions = args.serialize() + topo.serialize();

	cl::Program::Sources sources;
	sources.push_back({definitions.c_str(), definitions.length()});
	sources.push_back({TEMPORAL_SRC, strlen(TEMPORAL_SRC)});

	cl::Program program(context.nativeContext(), sources);
	try
	{
		program.build({context.device()});
	}
	catch(const cl::Error& err)
	{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.device()) << std::endl;
		throw;
	}

	m_timeStepKernel = cl::KernelFunctor(cl::Kernel(program, "timeStep"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_computeActiveStateKernel = cl::KernelFunctor(cl::Kernel(program, "computeActiveState"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_computePredictiveState = cl::KernelFunctor(cl::Kernel(program, "computePredictiveState"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_updateSynapsesKernel = cl::KernelFunctor(cl::Kernel(program, "updateSynapses"), context.queue(), cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	
	// Initialize region
	cl::KernelFunctor initRegion =
	cl::KernelFunctor(cl::Kernel(program, "initRegion"), context.queue(),
		cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);

	cl_uint2 randomState;
	randomState.s[0] = rand();
	randomState.s[1] = rand();
	initRegion(m_cellData.buffer(), m_segmentData.buffer(), m_synapseData.buffer(), randomState);
	std::cerr << "CLTemporalPooler: Kernels loaded" << std::endl;
}
void CLTemporalPooler::pullBuffers(bool cells, bool segments, bool synapses)
{
	if (cells)
		m_cellData.enqueueRead(false);
	if (segments)
		m_segmentData.enqueueRead(false);
	if (synapses)
		m_synapseData.enqueueRead(false);
	m_context.queue().finish();
}
void CLTemporalPooler::pushBuffers(bool cells, bool segments, bool synapses)
{
	if (cells)
		m_cellData.enqueueWrite(false);
	if (segments)
		m_segmentData.enqueueWrite(false);
	if (synapses)
		m_synapseData.enqueueWrite(false);
	m_context.queue().finish();
}

void CLTemporalPooler::write(const std::vector< cl_char >& activations_in, std::vector< cl_char >& results_out)
{
	if (activations_in.size() != std::size_t(m_topology.getColumns()))
	{
		throw std::runtime_error("Invalid vector length!");
	}

	// Send input column activations to device
	m_inputData.enqueueWrite(true, activations_in);

	// provide GPU some poor man's randomness
	cl_uint2 randomSeed;
	randomSeed.s[0] = rand();
	randomSeed.s[1] = rand();

	// Phase 0: Step forwards in time
	m_timeStepKernel(m_cellData.buffer(), m_segmentData.buffer(), m_synapseData.buffer());

	// Phase 1: Compute active state for each cell
	m_computeActiveStateKernel(m_cellData.buffer(), m_segmentData.buffer(), m_synapseData.buffer(), m_inputData.buffer(), randomSeed);

	// Phase 2: Compute predictive state for each cell
	m_computePredictiveState(m_cellData.buffer(), m_segmentData.buffer(), m_synapseData.buffer(), m_inputData.buffer(), randomSeed);

	// Phase 3: Update permanences
	m_updateSynapsesKernel(m_cellData.buffer(), m_segmentData.buffer(), m_synapseData.buffer(), m_inputData.buffer());

	// Obtain result (list of column activity) from compute device and save to results_out
	results_out.resize(m_topology.getColumns());
	m_inputData.enqueueRead(true, results_out);
}

void CLTemporalPooler::getStats(CLStats& stats)
{
	pullBuffers();

	stats.activeState = 0;
	stats.predictiveState = 0;
	stats.learningState = 0;
	for (int i = 0; i < int(m_cellData.size()); ++i)
	{
		CLCell& cell = m_cellData[i];
		if (cell.state & 0x1)
			stats.activeState ++;
		if (cell.state & 0x2)
			stats.predictiveState ++;
		if (cell.state & 0x4)
			stats.learningState ++;
	}

	stats.averageSegmentDutyCycle = 0;
// 	double act = 0;
	for (const CLSegment& seg: m_segmentData)
	{
		stats.averageSegmentDutyCycle += seg.activeDutyCycle;

// 		if (seg.activity[0][0] > act)
// 			act = seg.activity[0][0];
	}
	stats.averageSegmentDutyCycle /= m_segmentData.size();
// 	std::cout << "max segment activity: " << act << std::endl;

// 	double synsum = 0;
// 	for (const CLSynapse& syn: m_synapseData)
// 	{
// 		synsum += syn.permanence;
// 	}
// 	std::cout << (synsum / m_synapseData.size()) << std::endl;

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
