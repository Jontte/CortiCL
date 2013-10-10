#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <random>
#include <algorithm>
#include <sstream>

#include "clregion.h"

CLTemporalPooler::CLTemporalPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, const CLTopology& topo, const CLArgs& args)
	: m_device(device)
	, m_context(context)
	, m_commandQueue(queue)
	, m_topology(topo)
	, m_args(args)
	, m_cellDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(CLCell) * args.ColumnCellCount)
	, m_segmentDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(CLSegment) * args.ColumnCellCount * args.CellMaxSegments)
	, m_synapseDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(CLSynapse) * args.ColumnCellCount * args.CellMaxSegments * args.SegmentMaxSynapses)
	, m_inputDataBuffer(context, CL_MEM_READ_WRITE, m_topology.getColumns() * sizeof(cl_char))
{
	std::cerr << "CLTemporalPooler: Initializing" << std::endl;

	// Install kernel programs
	std::ifstream fin("cl/temporal.cl");
	std::string source = args.serialize() + std::string{std::istreambuf_iterator<char>(fin),std::istreambuf_iterator<char>()};
	cl::Program::Sources sources;
	sources.push_back({source.c_str(), source.length()});

	cl::Program program(context, sources);
	if (program.build({device}) != CL_SUCCESS)
	{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		throw std::runtime_error("Error compiling OpenCL source!");
	}

	m_timeStepKernel = cl::KernelFunctor(cl::Kernel(program, "timeStep"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_computeActiveStateKernel = cl::KernelFunctor(cl::Kernel(program, "computeActiveState"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_computePredictiveState = cl::KernelFunctor(cl::Kernel(program, "computePredictiveState"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);
	m_updateSynapsesKernel = cl::KernelFunctor(cl::Kernel(program, "updateSynapses"), m_commandQueue, cl::NullRange, cl::NDRange(m_topology.getColumns()), cl::NullRange);

	// Initialize all columns
	m_cellData.resize(m_topology.getColumns() * args.ColumnCellCount);
	m_segmentData.resize(m_topology.getColumns() * args.ColumnCellCount * args.CellMaxSegments);
	m_synapseData.resize(m_topology.getColumns() * args.ColumnCellCount * args.CellMaxSegments * args.SegmentMaxSynapses);

	std::random_device dev;
	std::mt19937 gen(dev());

	for (CLCell& cell: m_cellData)
	{
		cell.segmentCount = 0;
		cell.state = 0;
	}
	for (CLSegment& seg: m_segmentData)
	{
		seg.sequenceSegment = false;
		seg.hasQueuedChanges = false;
	}
	for (CLSynapse& syn: m_synapseData)
	{
		syn.permanence = 0;
		syn.targetCell = 0;
		syn.targetColumn = 0;
	}

	// Upload columns to GPU
	pushBuffers();
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
		m_commandQueue.enqueueWriteBuffer(m_cellDataBuffer, CL_FALSE, 0, m_cellData.size() * sizeof(CLCell), &m_cellData[0]);
	if (segments)
		m_commandQueue.enqueueWriteBuffer(m_segmentDataBuffer, CL_FALSE, 0, m_segmentData.size() * sizeof(CLSegment), &m_segmentData[0]);
	if (synapses)
		m_commandQueue.enqueueWriteBuffer(m_synapseDataBuffer, CL_FALSE, 0, m_synapseData.size() * sizeof(CLSynapse), &m_synapseData[0]);
	m_commandQueue.finish();
}

void CLTemporalPooler::write(const std::vector< cl_char >& activations_in, std::vector< cl_char >& results_out)
{
	if (activations_in.size() != std::size_t(m_topology.getColumns()))
	{
		throw std::runtime_error("Invalid vector length!");
	}

	cl_int err;

	err = m_commandQueue.enqueueWriteBuffer(m_inputDataBuffer, CL_FALSE, 0, m_topology.getColumns() * sizeof(cl_char), &activations_in[0]);
	if (err != CL_SUCCESS)
		throw std::runtime_error(getCLError(err));

	cl_uint2 randomSeed; // provide gpu some poor man's randomness
	randomSeed.s[0] = rand();
	randomSeed.s[1] = rand();

	// Phase 0: Step forwards in time
	m_timeStepKernel(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer);
	err = m_timeStepKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("timeStepKernel: " + getCLError(err));

	// Phase 1: Compute active state for each cell
	m_computeActiveStateKernel(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer, m_inputDataBuffer, randomSeed);
	err = m_computeActiveStateKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("computeActiveStateKernel: " + getCLError(err));

	// Phase 2: Compute predictive state for each cell
	m_computePredictiveState(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer, m_inputDataBuffer, randomSeed);
	err = m_computePredictiveState.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("computePredictiveStateKernel: " + getCLError(err));

	// Phase 3: Update permanences
	m_updateSynapsesKernel(m_cellDataBuffer, m_segmentDataBuffer, m_synapseDataBuffer, m_inputDataBuffer);
	err = m_updateSynapsesKernel.getError();
	if (err != CL_SUCCESS)
		throw std::runtime_error("updateSynapsesKernel: " + getCLError(err));

	results_out.resize(m_topology.getColumns());
	m_commandQueue.enqueueReadBuffer(m_inputDataBuffer, CL_TRUE, 0, sizeof(cl_char) * m_topology.getColumns(), &results_out[0]);
}

void CLTemporalPooler::getStats(CLStats& stats)
{
	pullBuffers();

	stats.maxSegments = m_topology.getColumns() * m_args.ColumnCellCount * m_args.CellMaxSegments;
	stats.maxSynapses = m_topology.getColumns() * m_args.ColumnCellCount * m_args.CellMaxSegments * m_args.SegmentMaxSynapses;
	stats.totalSegments = 0;
	stats.totalSynapses = 0;

	for (int i = 0; i < m_cellData.size(); ++i)
	{
		CLCell& cell = m_cellData[i];
		stats.totalSegments += cell.segmentCount;

		for (int a = 0; a < cell.segmentCount; ++a)
		{
			int offset = i * m_args.CellMaxSegments;
			CLSegment& seg = m_segmentData[offset+a];
			stats.totalSynapses += seg.synapseCount;
		}
	}
}
