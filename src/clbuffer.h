#ifndef CLBUFFER_H_INCLUDED
#define CLBUFFER_H_INCLUDED

#include "clcontext.h"
#include <cassert>
#include <vector>

template <class T>
class CLBuffer
{
private:
	CLContext& m_context;
	cl::Buffer m_buffer;
	std::vector<T> m_data;
	std::size_t m_byteSize;
public:

	CLBuffer(CLContext& context, std::size_t length)
		: m_context(context)
		, m_buffer(context.nativeContext(), CL_MEM_READ_WRITE, length * sizeof(T))
		, m_data(length)
		, m_byteSize(length * sizeof(T))
	{}

	cl::Buffer& buffer() { return m_buffer; }

	// Write data to device side buffer
	void enqueueWrite(bool blocking)
	{
		enqueueWrite(blocking, m_data);
	}
	// Write data to device side buffer from external source
	void enqueueWrite(bool blocking, const std::vector<T>& data)
	{
		assert(data.size() == m_data.size());
		assert(data.size() * sizeof(T) == m_byteSize);
		m_context.queue().enqueueWriteBuffer(m_buffer, blocking ? CL_TRUE : CL_FALSE, 0, m_byteSize, &data[0]);
	}
	// Read data from device
	void enqueueRead(bool blocking)
	{
		enqueueRead(blocking, m_data);
	}
	// Read data from device to external buffer
	void enqueueRead(bool blocking, std::vector<T>& data)
	{
		assert(data.size() == m_data.size());
		assert(data.size() * sizeof(T) == m_byteSize);
		m_context.queue().enqueueReadBuffer(m_buffer, blocking ? CL_TRUE : CL_FALSE, 0, m_byteSize, &data[0]);
	}

	// Define some accessors to the underlying std::vector
	inline typename std::vector<T>::iterator begin() { return m_data.begin(); }
	inline typename std::vector<T>::iterator end  () { return m_data.end();   }

	inline T& operator[](std::size_t index)
	{
		return m_data[index];
	}
	inline std::size_t size() const { return m_data.size(); }
};

#endif
