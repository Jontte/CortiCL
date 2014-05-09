#ifndef CLCONTEXT_H_DEFINED
#define CLCONTEXT_H_DEFINED

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

class CLContext
{
private:
	cl::Device m_device;
	cl::Context m_context;
	cl::CommandQueue m_queue;

public:
	CLContext();

	cl::Device& device() { return m_device; }
	cl::Context& nativeContext() { return m_context; }
	cl::CommandQueue& queue() { return m_queue; }
};

#endif
