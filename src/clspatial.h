#ifndef CLSPATIAL_H_INCLUDED
#define CLSPATIAL_H_INCLUDED

#include <string>
#include <CL/cl.hpp>

std::string getCLError(cl_int err);

struct CLStats;
class CLSpatialPooler
{
private:

	constexpr static int ColumnSynapses = 10;

	struct CLSynapse
	{
		cl_float permanence;
		cl_int target;
	};
	struct CLColumn
	{
		CLSynapse proximalSynapses[ColumnSynapses];
		cl_float boost;
		cl_float overlap;
		cl_bool active;

		cl_float activeDutyCycle;
		cl_float minDutyCycle;
		cl_float overlapDutyCycle;
	};

	cl::Device& m_device;
	cl::Context& m_context;
	cl::CommandQueue& m_commandQueue;

	int m_columns;
	int m_inputSize;

	cl::KernelFunctor m_computeOverlapKernel;
	cl::KernelFunctor m_inhibitNeighboursKernel;
	cl::KernelFunctor m_updatePermanencesKernel;

	std::vector<CLColumn> m_columnData;
	cl::Buffer m_columnDataBuffer;
	cl::Buffer m_inputDataBuffer;

public:

	CLSpatialPooler(cl::Device& device, cl::Context& context, cl::CommandQueue& queue, int columns, int inputSize);
	std::vector<cl_char> write(const std::vector<cl_char>& bits);
	void getStats(CLStats& stats);
};


#endif
