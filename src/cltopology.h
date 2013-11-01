#ifndef CLTOPOLOGY_H_INCLUDED
#define CLTOPOLOGY_H_INCLUDED

#include <string>

struct CLTopology
{
	// Input data topology
	int inputWidth;
	int inputHeight;

	// Column topology
	int regionWidth;
	int regionHeight;

	// How far the column's neighbourhood spans or -1 for global inhibition
	int inhibitionRadius;

	// How far columns extend their receptive field in the input space or -1 for unlimited
	int receptiveFieldRadius;

	inline int getInputSize() const { return inputWidth * inputHeight; }
	inline int getColumns() const { return regionWidth * regionHeight; }

	static CLTopology globalInhibition2D(int inputWidth, int inputHeight, int regionWidth, int regionHeight)
	{
		CLTopology ret;
		ret.inputWidth = inputWidth;
		ret.inputHeight = inputHeight;
		ret.regionWidth = regionWidth;
		ret.regionHeight = regionHeight;
		ret.inhibitionRadius = -1;
		ret.receptiveFieldRadius = -1;
		return ret;
	}
	static CLTopology localInhibition2D(int inputWidth, int inputHeight, int regionWidth, int regionHeight, int inhibitionRadius, int receptiveFieldRadius)
	{
		CLTopology ret;
		ret.inputWidth = inputWidth;
		ret.inputHeight = inputHeight;
		ret.regionWidth = regionWidth;
		ret.regionHeight = regionHeight;
		ret.inhibitionRadius = inhibitionRadius;
		ret.receptiveFieldRadius = receptiveFieldRadius;
		return ret;
	}

	static CLTopology line(int inputLength, int regionLength, int inhibitionRadius, int receptiveFieldRadius)
	{
		CLTopology ret;
		ret.inputWidth = inputLength;
		ret.inputHeight = 1;
		ret.regionWidth = regionLength;
		ret.regionHeight = 1;
		ret.inhibitionRadius = inhibitionRadius;
		ret.receptiveFieldRadius = receptiveFieldRadius;
		return ret;
	}
	
	std::string serialize() const;
};


#endif
