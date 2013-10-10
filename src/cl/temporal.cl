typedef enum {NOW, WAS} TimeStep;

typedef enum {
	ACTIVESTATE = 0x01,
	PREDICTIVESTATE = 0x02,
	LEARNSTATE = 0x04
} CellState;


typedef struct
{
	float permanence;
	float permanenceQueued; // segment updates from SegmentUpdate structures is flattened here
	int targetColumn;
	uchar targetCell;
	uchar targetCellState;
} Synapse;

typedef struct
{
	Synapse synapses[MaxSynapses];
	uchar synapseCount;

	// Activity of the segment
	// 0 = activeState, 1 = learnState
	// 0 = now, 1 = previous timestep
	uchar activity[2][2];

	// Activity that includes synapses with permanence below CONNECTED_PERMANENCE but above MIN_PERMANENCE
	// 0 = activeState, 1 = learnState
	// 0 = now, 1 = previous timestep
	uchar fullActivity[2][2];
	bool sequenceSegment;
	bool sequenceSegmentQueued;
	bool hasQueuedChanges;
} Segment;

typedef struct
{
	Segment segments[MaxSegments];
	uchar segmentCount;
	uchar newSegmentCount;
	uchar state;
} Cell;


typedef struct
{
	Cell cells[ColumnCells];
} Column;

inline bool getState(uchar state, TimeStep when, uchar stateMask)
{
	return state & (stateMask << (when*4));
}
inline bool cellState(global Cell* cell, TimeStep when, uchar stateMask)
{
	return getState(cell->state, when, stateMask);
}
inline void setCell(global Cell* cell, uchar stateMask)
{
	cell->state |= stateMask;
}

uint random(uint2 seedValue)
{
	uint seed = seedValue.x + get_global_id(0);
	uint t = seed ^ (seed << 11);
	return seedValue.y ^ (seedValue.y >> 19) ^ (t ^ (t >> 8));
}

bool segmentActivity(global Segment* segment, TimeStep when, CellState state)
{
	if (state == ACTIVESTATE)
		return segment->activity[0][when];
	if (state == LEARNSTATE)
		return segment->activity[1][when];
	return 0;
}
bool segmentActive(global Segment* segment, TimeStep when, CellState state)
{
	return segmentActivity(segment, when, state) > SEGMENT_ACTIVATION_THRESHOLD;
}


global Segment* getActiveSegment(global Cell* cell, TimeStep when, CellState state)
{
	bool activeSequenceSegments = false;
	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* seg = &cell->segments[i];
		if (seg->sequenceSegment && segmentActive(seg, when, state))
		{
			activeSequenceSegments = true;
			break;
		}
	}

	int bestActivityIdx = -1;
	int bestActivity = -1;

	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* seg = &cell->segments[i];
		if (activeSequenceSegments && !seg->sequenceSegment)
			continue;

		int act = segmentActivity(seg, when, state);
		if (act > SEGMENT_MIN_THRESHOLD && act > bestActivity)
		{
			bestActivity = act;
			bestActivityIdx = i;
		}
	}
	if (bestActivityIdx != -1)
		return &cell->segments[bestActivityIdx];
	return 0;
}


typedef struct
{
	int activity;
	global Segment* segment;
} BestMatchingSegmentStruct;

BestMatchingSegmentStruct getBestMatchingSegment(global Column* col, global Cell* cell, TimeStep when)
{
	BestMatchingSegmentStruct ret;
	ret.activity = 0;
	ret.segment = 0;

	int bestActivityIdx = -1;
	int bestActivity = 0;

	// Return segment with highest activity

	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* segment = &cell->segments[i];

		// Check synapses that are not even fully connected
		int activity = segment->fullActivity[ACTIVESTATE][when];

		if (activity > bestActivity)
		{
			bestActivity = activity;
			bestActivityIdx = i;
		}
	}
	ret.activity = bestActivity;
	ret.segment = &cell->segments[bestActivityIdx];
	return  ret;
}

typedef struct
{
	global Cell* cell;
	global Segment* segment;
} BestMatchingCellStruct;

BestMatchingCellStruct getBestMatchingCell(global Column* col, TimeStep when)
{
	BestMatchingCellStruct ret;
	ret.cell = 0;
	ret.segment = 0;

	{
		int bestActivityIdx = -1;
		int bestActivity = 0;

		// Return cell and segment with highest activity

		for (int i = 0; i < ColumnCells; ++i)
		{
			global Cell* cell = &col->cells[i];
			BestMatchingSegmentStruct bestSegment = getBestMatchingSegment(col, cell, when);

			if (bestSegment.activity > bestActivity)
			{
				bestActivity = bestSegment.activity;
				ret.cell = &col->cells[i];
				ret.segment = bestSegment.segment;
			}
		}

		if (bestActivityIdx != -1)
		{
			return ret;
		}
	}

	// No best cell, return just cell with fewest segments
	int fewestSegmentsIdx = -1;
	int fewestSegments = 0;

	for (int i = 0; i < ColumnCells; ++i)
	{
		global Cell* cell = &col->cells[i];
		int num = cell->segmentCount;

		if (num < fewestSegments || fewestSegmentsIdx == -1)
		{
			fewestSegments = num;
			fewestSegmentsIdx = i;
		}
	}
	ret.cell = &col->cells[fewestSegmentsIdx];
	ret.segment = 0;
	return ret;

}

void getSegmentActiveSynapses(
	global Column* allColumns,
	const int columnId,
	global Cell* cell,
	global Segment* segment,
	TimeStep when,
	bool newSynapses,
	uint2 seedValue // Seed value provided by cpu
	)
{
	// Add a segment update structure to given cell
	if (!segment)
	{
		// Create new segment
		if (cell->segmentCount + cell->newSegmentCount >= MaxSegments)
		{
			// No more room in this cell..
			return;
		}

		segment = &cell->segments[cell->segmentCount + cell->newSegmentCount];
		cell->newSegmentCount++;

		segment->synapseCount = 0;
		segment->activity[0][0] = 0;
		segment->activity[1][0] = 0;
		segment->activity[0][1] = 0;
		segment->activity[1][1] = 0;
		segment->fullActivity[0][0] = 0;
		segment->fullActivity[1][0] = 0;
		segment->fullActivity[0][1] = 0;
		segment->fullActivity[1][1] = 0;
		segment->sequenceSegment = 0;
		segment->hasQueuedChanges = false;
	}

	// If no changes have been queued, set permamenceQueued of each synapse to match current permanence
	bool hasChanges = segment->hasQueuedChanges;
	if (!hasChanges)
	{
		for (int i = 0 ; i < segment->synapseCount; ++i)
		{
			global Synapse* synapse = &segment->synapses[i];
			synapse->permanenceQueued = synapse->permanence;
		}
		segment->sequenceSegmentQueued = segment->sequenceSegment;
		segment->hasQueuedChanges = true;
	}

	// Find active synapses in segment
	for (int i = 0 ; i < segment->synapseCount; ++i)
	{
		global Synapse* synapse = &segment->synapses[i];
		if (getState(synapse->targetCellState, when, ACTIVESTATE))
		{
			synapse->permanenceQueued += PERMANENCE_STEP;
		}
		else
		{
			synapse->permanenceQueued -= PERMANENCE_STEP;
		}
	}

	// Enhance segment by adding new synapses
	if (newSynapses)
	{
		int insertId = 0;
		if (segment->synapseCount >= MaxSynapses)
		{
			// Drop one inactive synapse to make room..
			for (int i = 0; i < segment->synapseCount; ++i)
			{
				global Synapse* synapse = &segment->synapses[i];
				if (! getState(synapse->targetCellState, when, LEARNSTATE))
				{
					insertId = i;
					break;
				}
			}
		}
		else
		{
			insertId = segment->synapseCount;
		}

		// Globally search for a learning cell. SLOW.
		// Generate a random number to decide where to start the search

		int randOffset = random(seedValue);

		int targetColumn = -1;
		int targetCell = -1;
		for (int i = 0; i < get_global_size(0); ++i)
		{
			int testColumnId = (columnId+i) % get_global_size(0);
			global Column* testColumn = &allColumns[testColumnId];

			if (testColumnId == columnId)
				continue;

			for (int a = 0; a < ColumnCells; ++a)
			{
				global Cell* testCell = &testColumn->cells[a];
				if ( cellState(testCell, when, LEARNSTATE))
				{
					// Found one!
					targetColumn = testColumnId;
					targetCell = a;
					break;
				}
			}
			if (targetColumn != -1)
				break;
		}

		if (targetColumn != -1)
		{
			// We have a new target for the synapse
			if (segment->synapseCount < MaxSynapses)
				segment->synapseCount++;

			global Synapse* syn = &segment->synapses[insertId];
			syn->targetColumn = targetColumn;
			syn->targetCell = targetCell;
			syn->permanence = 0.2;
			syn->permanenceQueued = 0.2;
			syn->targetCellState = 0;
		}
	}
}

void adaptSegments(global Cell* cell, bool positiveReinforcement)
{
	cell->segmentCount += cell->newSegmentCount;
	cell->newSegmentCount = 0;

	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* segment = &cell->segments[i];

		if (!segment->hasQueuedChanges)
			continue;
		segment->hasQueuedChanges = false;
		segment->sequenceSegment = segment->sequenceSegmentQueued;

		if (positiveReinforcement)
		{
			// Cool, use the enhanced values of all synapses
			for (int a = 0; a < segment->synapseCount; ++a)
			{
				global Synapse* synapse = &segment->synapses[a];
				synapse->permanence = synapse->permanenceQueued;
			}
		}
		else
		{
			// Oops, misprediction. Synapses with enhanced permanence get their permanence flipped, the rest stay untouched

			for (int a = 0; a < segment->synapseCount; ++a)
			{
				global Synapse* synapse = &segment->synapses[a];

				if (synapse->permanenceQueued > synapse->permanence)
				{
					synapse->permanence += synapse->permanence - synapse->permanenceQueued;
				}
			}
		}
		for (int a = 0; a < segment->synapseCount; ++a)
		{
			global Synapse* synapse = &segment->synapses[a];

			if (synapse->permanence > 1.0)
				synapse->permanence = 1.0;
			else if (synapse->permanence < 0.0)
				synapse->permanence = 0.0;
		}
	}
}

void kernel timeStep(global Column* columns)
{
	int index = get_global_id(0);
	global Column* col = &columns[index];

	for (int i = 0 ; i < ColumnCells; ++i)
	{
		global Cell* cell = &col->cells[i];
		cell->state = cell->state << 4;

		for (int a = 0; a < cell->segmentCount; ++a)
		{
			global Segment* segment = &cell->segments[a];
			segment->activity[0][WAS] = segment->activity[0][NOW];
			segment->activity[1][WAS] = segment->activity[1][NOW];
			segment->activity[0][NOW] = 0;
			segment->activity[1][NOW] = 0;
			segment->fullActivity[0][WAS] = segment->fullActivity[0][NOW];
			segment->fullActivity[1][WAS] = segment->fullActivity[1][NOW];
			segment->fullActivity[0][NOW] = 0;
			segment->fullActivity[1][NOW] = 0;

			for (int b = 0; b < segment->synapseCount; ++b)
			{
				global Synapse* synapse = &segment->synapses[b];
				synapse->targetCellState = synapse->targetCellState << 4;
			}
		}
	}
}

void kernel computeActiveState(global Column* columns, global const char* activeColumns, uint2 seedValue)
{
	int index = get_global_id(0);

	if (!activeColumns[index])
		return;

	global Column* col = &columns[index];

	bool buPredicted = false;
	bool lcChosen = false;

	// Check if any cell predicted this column activation
	for (int i = 0 ; i < ColumnCells; ++i)
	{
		global Cell* cell = &col->cells[i];

		if (cellState(cell, WAS, PREDICTIVESTATE))
		{
			global Segment* segment = getActiveSegment(cell, WAS, ACTIVESTATE);
			if (!segment)
			{
				// We shouldn't end up here...
				return;
			}
			if (segment->sequenceSegment)
			{
				buPredicted = true;
				setCell(cell, ACTIVESTATE);
				if (segmentActive(segment, WAS, LEARNSTATE))
				{
					lcChosen = true;
					setCell(cell, LEARNSTATE);
				}
			}
		}
	}

	if (!buPredicted)
	{
		// Bottom-up input was unexpected -> activate all cells
		for (int i = 0 ; i < ColumnCells; ++i)
		{
			global Cell* cell = &col->cells[i];
			setCell(cell, ACTIVESTATE);
		}
	}
	if (!lcChosen)
	{
		BestMatchingCellStruct ret = getBestMatchingCell(col, WAS);
		global Cell* learnCell = ret.cell;
		global Segment* learnSegment = ret.segment;
		setCell(learnCell, LEARNSTATE);

		getSegmentActiveSynapses(columns, index, learnCell, learnSegment, WAS, true, seedValue);
		if (learnSegment)
			learnSegment->sequenceSegmentQueued = true;
	}
}

void kernel computePredictiveState(global Column* columns, global const char* activeColumns, uint2 seedValue)
{
	int index = get_global_id(0);
	global Column* col = &columns[index];

	for (int i = 0 ; i < ColumnCells; ++i)
	{
		global Cell* cell = &col->cells[i];

		int segments = cell->segmentCount;
		for (int a = 0 ; a < segments; ++a)
		{
			global Segment* segment = &cell->segments[a];

			// Cache segment activity here..

			int activity = 0;
			int fullActivity = 0;
			int learnActivity = 0;
			int fullLearnActivity = 0;
			for (int b = 0 ; b < segment->synapseCount; ++b)
			{
				global Synapse* syn = &segment->synapses[b];
				global Column* targetColumn = &columns[syn->targetColumn];
				global Cell* targetCell = &targetColumn->cells[syn->targetCell];
				syn->targetCellState |= targetCell->state & 0xF;

				if (cellState(targetCell, NOW, ACTIVESTATE))
				{
					fullActivity++;
					if (syn->permanence > CONNECTED_PERMANENCE)
						activity++;
				}
				if (cellState(targetCell, NOW, LEARNSTATE))
				{
					fullLearnActivity++;
					if (syn->permanence > CONNECTED_PERMANENCE)
						learnActivity++;
				}
			}
			segment->activity[0][NOW] = activity;
			segment->fullActivity[0][NOW] = fullActivity;
			segment->activity[1][NOW] = learnActivity;
			segment->fullActivity[1][NOW] = fullLearnActivity;

			if (segmentActive(segment, NOW, ACTIVESTATE))
			{
				setCell(cell, PREDICTIVESTATE);

				getSegmentActiveSynapses(columns, index, cell, segment, NOW, false, seedValue);

				BestMatchingSegmentStruct bestMatch = getBestMatchingSegment(col, cell, WAS);
				getSegmentActiveSynapses(columns, index, cell, bestMatch.segment, WAS, true, seedValue);
			}
		}
	}
}

void kernel updateSynapses(global Column* columns, global char* resultBuffer)
{
	int index = get_global_id(0);
	global Column* col = &columns[index];
	global char* result = resultBuffer + index;

	bool columnActive = false;
	for (int i = 0 ; i < ColumnCells; ++i)
	{
		global Cell* cell = &col->cells[i];

		if (cellState(cell, NOW, LEARNSTATE))
		{
			adaptSegments(cell, true);
		}
		else if(!cellState(cell, NOW, PREDICTIVESTATE) && cellState(cell, WAS, PREDICTIVESTATE))
		{
			adaptSegments(cell, false);
		}
		if (cellState(cell, NOW, ACTIVESTATE|PREDICTIVESTATE))
		{
			columnActive = true;
		}
	}
	*result = columnActive;
}
