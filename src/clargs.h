#ifndef CLARGS_H_INCLUDED
#define CLARGS_H_INCLUDED

#include <string>

struct CLArgs
{
	float ConnectedPermanence = 0.2;
	float PermanenceStep = 0.05;

	// Spatial pooler:
	int ColumnProximalSynapseCount = 10;
	int ColumnProximalSynapseMinOverlap = 7;
	float BoostStep = 0.01;
	float DutyCyclePersistence = 0.99;

	// Temporal pooler:
	int ColumnCellCount = 4;
	int CellMaxSegments = 10;
	int SegmentMaxSynapses = 10;
	int SegmentActivationThreshold = 5;
	int SegmentMinThreshold = 3;

	std::string serialize() const;
};

#endif
