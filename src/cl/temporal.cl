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
	uchar segmentCount;
	uchar newSegmentCount;
	uchar state;
} Cell;

typedef struct
{
	global Cell* cells;
	global Segment* segments;
	global Synapse* synapses;
} State;

State makeState(global Cell* cells, global Segment* segments, global Synapse* synapses)
{
	State ret;
	ret.cells = cells;
	ret.segments = segments;
	ret.synapses = synapses;
	return ret;
}

inline global Cell* getCells(const State* state, int columnIdx)
{
	return &state->cells[columnIdx * COLUMN_CELL_COUNT];
}
inline global Segment* getSegments(const State* state, int columnIdx, int cellIdx)
{
	return &state->segments[
	columnIdx * CELL_MAX_SEGMENTS * COLUMN_CELL_COUNT
	+ cellIdx * CELL_MAX_SEGMENTS];
}
inline global Synapse* getSynapses(const State* state, int columnIdx, int cellIdx, int segmentIdx)
{
	return &state->synapses[
	columnIdx * SEGMENT_MAX_SYNAPSES * CELL_MAX_SEGMENTS * COLUMN_CELL_COUNT
	+ cellIdx * SEGMENT_MAX_SYNAPSES * CELL_MAX_SEGMENTS
	+ segmentIdx * SEGMENT_MAX_SYNAPSES
	];
}


inline bool getCellState(uchar state, TimeStep when, uchar stateMask)
{
	return state & (stateMask << (when*4));
}
inline void setCellState(global Cell* cell, uchar stateMask)
{
	cell->state |= stateMask;
}

uint random(uint2 seedValue)
{
	uint seed = seedValue.x + get_global_id(0);
	uint t = seed ^ (seed << 11);
	return seedValue.y ^ (seedValue.y >> 19) ^ (t ^ (t >> 8));
}

inline bool segmentActivity(global Segment* segment, TimeStep when, CellState state)
{
	if (state == ACTIVESTATE)
		return segment->activity[0][when];
	if (state == LEARNSTATE)
		return segment->activity[1][when];
	return 0;
}
inline bool segmentActive(global Segment* segment, TimeStep when, CellState state)
{
	return segmentActivity(segment, when, state) > SEGMENT_ACTIVATION_THRESHOLD;
}

global Segment* getActiveSegment(const State* state, int columnIdx, int cellIdx, TimeStep when, CellState cellState)
{
	global Cell* cell = getCells(state, columnIdx) + cellIdx;
	global Segment* segments = getSegments(state, columnIdx, cellIdx);

	bool activeSequenceSegments = false;
	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* seg = segments + i;
		if (seg->sequenceSegment && segmentActive(seg, when, cellState))
		{
			activeSequenceSegments = true;
			break;
		}
	}

	int bestActivityIdx = -1;
	int bestActivity = -1;

	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* seg = segments + i;
		if (activeSequenceSegments && !seg->sequenceSegment)
			continue;

		int act = segmentActivity(seg, when, cellState);
		if (act > SEGMENT_MIN_THRESHOLD && act > bestActivity)
		{
			bestActivity = act;
			bestActivityIdx = i;
		}
	}
	if (bestActivityIdx != -1)
		return segments + bestActivityIdx;
	return 0;
}

typedef struct
{
	int activity;
	global Segment* segment;
	int segmentIdx;
} BestMatchingSegmentStruct;

BestMatchingSegmentStruct getBestMatchingSegment(const State* state, int columnIdx, int cellIdx, TimeStep when)
{
	global Cell* cell = getCells(state, columnIdx) + cellIdx;
	BestMatchingSegmentStruct ret;
	ret.activity = 0;
	ret.segment = 0;
	ret.segmentIdx = -1;

	int bestActivityIdx = -1;
	int bestActivity = 0;

	// Return segment with highest activity

	global Segment* segments = getSegments(state, columnIdx, cellIdx);
	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* seg = segments + i;

		// Check synapses that are not even fully connected
		int activity = seg->fullActivity[ACTIVESTATE][when];

		if (activity > bestActivity)
		{
			bestActivity = activity;
			bestActivityIdx = i;
		}
	}
	ret.activity = bestActivity;
	ret.segment = segments + bestActivityIdx;
	ret.segmentIdx = bestActivityIdx;
	return  ret;
}

typedef struct
{
	global Cell* cell;
	int cellIdx;
	global Segment* segment;
	int segmentIdx;
} BestMatchingCellStruct;

BestMatchingCellStruct getBestMatchingCell(const State* state, int columnIdx, TimeStep when)
{
	global Cell* cells = getCells(state, columnIdx);
	BestMatchingCellStruct ret;
	ret.cell = 0;
	ret.segment = 0;
	ret.cellIdx = -1;
	ret.segmentIdx = -1;

	{
		int bestActivityIdx = -1;
		int bestActivity = 0;

		// Return cell and segment with highest activity

		for (int i = 0; i < COLUMN_CELL_COUNT; ++i)
		{
			global Cell* cell = cells + i;
			BestMatchingSegmentStruct bestSegment =
				getBestMatchingSegment(state, columnIdx, i, when);

			if (bestSegment.activity > bestActivity)
			{
				bestActivity = bestSegment.activity;
				ret.cell = cell;
				ret.cellIdx = i;
				ret.segment = bestSegment.segment;
				ret.segmentIdx = bestSegment.segmentIdx;
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

	for (int i = 0; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;
		int num = cell->segmentCount;

		if (num < fewestSegments || fewestSegmentsIdx == -1)
		{
			fewestSegments = num;
			fewestSegmentsIdx = i;
		}
	}
	ret.cell = cells + fewestSegmentsIdx;
	ret.cellIdx = fewestSegmentsIdx;
	ret.segment = 0;
	ret.segmentIdx = -1;
	return ret;
}

void getSegmentActiveSynapses(
	const State* state,
	int columnIdx,
	int cellIdx,
	int segmentIdx,
	TimeStep when,
	bool newSynapses,
	uint2 seedValue // Seed value provided by cpu
	)
{
	global Cell* cell = getCells(state, columnIdx) + cellIdx;
	global Segment* segment = 0;

	// Add a segment update structure to given cell
	if (segmentIdx == -1)
	{
		// Create new segment
		if (cell->segmentCount + cell->newSegmentCount >= CELL_MAX_SEGMENTS)
		{
			// No more room in this cell..
			return;
		}

		segmentIdx = cell->segmentCount + cell->newSegmentCount;
		segment = getSegments(state, columnIdx, cellIdx) + segmentIdx;
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
	else
	{
		segment = getSegments(state, columnIdx, cellIdx) + segmentIdx;
	}

	global Synapse* synapses = getSynapses(state, columnIdx, cellIdx, segmentIdx);

	// If no changes have been queued, set permamenceQueued of each synapse to match current permanence
	bool hasChanges = segment->hasQueuedChanges;
	if (!hasChanges)
	{
		for (int i = 0 ; i < segment->synapseCount; ++i)
		{
			global Synapse* synapse = synapses + i;
			synapse->permanenceQueued = synapse->permanence;
		}
		segment->sequenceSegmentQueued = segment->sequenceSegment;
		segment->hasQueuedChanges = true;
	}

	// Find active synapses in segment
	for (int i = 0 ; i < segment->synapseCount; ++i)
	{
		global Synapse* synapse = synapses + i;
		if (getCellState(synapse->targetCellState, when, ACTIVESTATE))
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
		if (segment->synapseCount >= SEGMENT_MAX_SYNAPSES)
		{
			// Drop one inactive synapse to make room..
			for (int i = 0; i < segment->synapseCount; ++i)
			{
				global Synapse* synapse = synapses + i;
				if (! getCellState(synapse->targetCellState, when, LEARNSTATE))
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
			int testColumnId = (columnIdx+i) % get_global_size(0);

			global Cell* testColumn = getCells(state, testColumnId);

			if (testColumnId == columnIdx) // Skip self...
				continue;

			for (int a = 0; a < COLUMN_CELL_COUNT; ++a)
			{
				global Cell* testCell = testColumn + a;
				if (getCellState(testCell->state, when, LEARNSTATE))
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
			if (segment->synapseCount < SEGMENT_MAX_SYNAPSES)
				segment->synapseCount++;

			global Synapse* syn = getSynapses(state, columnIdx, cellIdx, segmentIdx) + insertId;
			syn->targetColumn = targetColumn;
			syn->targetCell = targetCell;
			syn->permanence = 0.2;
			syn->permanenceQueued = 0.2;
			syn->targetCellState = 0;
		}
	}
}

void adaptSegments(const State* state, int columnIdx, int cellIdx, bool positiveReinforcement)
{
	global Cell* cell = getCells(state, columnIdx) + cellIdx;
	cell->segmentCount += cell->newSegmentCount;
	cell->newSegmentCount = 0;

	global Segment* segments = getSegments(state, columnIdx, cellIdx);
	for (int i = 0; i < cell->segmentCount; ++i)
	{
		global Segment* segment = segments + i;
		global Synapse* synapses = getSynapses(state, columnIdx, cellIdx, i);

		if (!segment->hasQueuedChanges)
			continue;
		segment->hasQueuedChanges = false;
		segment->sequenceSegment = segment->sequenceSegmentQueued;

		if (positiveReinforcement)
		{
			// Cool, use the enhanced values of all synapses
			for (int a = 0; a < segment->synapseCount; ++a)
			{
				global Synapse* synapse = synapses + a;
				synapse->permanence = synapse->permanenceQueued;
			}
		}
		else
		{
			// Oops, misprediction. Synapses with enhanced permanence get their permanence flipped, the rest stay untouched

			for (int a = 0; a < segment->synapseCount; ++a)
			{
				global Synapse* synapse = synapses + a;
				if (synapse->permanenceQueued > synapse->permanence)
				{
					synapse->permanence += synapse->permanence - synapse->permanenceQueued;
				}
			}
		}
		for (int a = 0; a < segment->synapseCount; ++a)
		{
			global Synapse* synapse = synapses + a;
			if (synapse->permanence > 1.0)
				synapse->permanence = 1.0;
			else if (synapse->permanence < 0.0)
				synapse->permanence = 0.0;
		}
	}
}

void kernel timeStep(
	global Cell* g_cells,
	global Segment* g_segments,
	global Synapse* g_synapses)
{
	State state = makeState(g_cells, g_segments, g_synapses);
	int columnIdx = get_global_id(0);

	// Get cells of the current column
	global Cell* cells = getCells(&state, columnIdx);

	for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;
		cell->state = cell->state << 4;

		global Segment* segments = getSegments(&state, columnIdx, i);
		for (int a = 0; a < cell->segmentCount; ++a)
		{
			global Segment* segment = segments + a;
			segment->activity[0][WAS] = segment->activity[0][NOW];
			segment->activity[1][WAS] = segment->activity[1][NOW];
			segment->activity[0][NOW] = 0;
			segment->activity[1][NOW] = 0;
			segment->fullActivity[0][WAS] = segment->fullActivity[0][NOW];
			segment->fullActivity[1][WAS] = segment->fullActivity[1][NOW];
			segment->fullActivity[0][NOW] = 0;
			segment->fullActivity[1][NOW] = 0;

			global Synapse* synapses = getSynapses(&state, columnIdx, i, a);
			for (int b = 0; b < segment->synapseCount; ++b)
			{
				global Synapse* synapse = synapses + b;
				synapse->targetCellState = synapse->targetCellState << 4;
			}
		}
	}
}

void kernel computeActiveState(
	global Cell* g_cells,
	global Segment* g_segments,
	global Synapse* g_synapses,
	global const char* activeColumns,
	uint2 seedValue)
{
	State state = makeState(g_cells, g_segments, g_synapses);
	int columnIdx = get_global_id(0);

	if (!activeColumns[columnIdx])
		return;

	global Cell* cells = getCells(&state, columnIdx);

	bool buPredicted = false;
	bool lcChosen = false;

	// Check if any cell predicted this column activation
	for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;

		if (getCellState(cell->state, WAS, PREDICTIVESTATE))
		{
			global Segment* segment = getActiveSegment(&state, columnIdx, i, WAS, ACTIVESTATE);
			if (!segment)
			{
				// We shouldn't end up here...
				return;
			}
			if (segment->sequenceSegment)
			{
				buPredicted = true;

				setCellState(cell, ACTIVESTATE);
				if (segmentActive(segment, WAS, LEARNSTATE))
				{
					lcChosen = true;
					setCellState(cell, LEARNSTATE);
				}
			}
		}
	}

	if (!buPredicted)
	{
		// Bottom-up input was unexpected -> activate all cells
		global Cell* cells = getCells(&state, columnIdx);
		for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
		{
			global Cell* cell = cells + i;
			setCellState(cell, ACTIVESTATE);
		}
	}
	if (!lcChosen)
	{
		BestMatchingCellStruct ret = getBestMatchingCell(&state, columnIdx, WAS);
		global Cell* learnCell = ret.cell;
		int learnCellIdx = ret.cellIdx;
		global Segment* learnSegment = ret.segment;
		int learnSegmentIdx = ret.segmentIdx;
		setCellState(learnCell, LEARNSTATE);

		getSegmentActiveSynapses(&state, columnIdx, learnCellIdx, learnSegmentIdx, WAS, true, seedValue);
		if (learnSegment)
			learnSegment->sequenceSegmentQueued = true;
	}
}

void kernel computePredictiveState(
	global Cell* g_cells,
	global Segment* g_segments,
	global Synapse* g_synapses,
	global const char* activeColumns,
	uint2 seedValue)
{
	State state = makeState(g_cells, g_segments, g_synapses);

	int columnIdx = get_global_id(0);
	global Cell* cells = getCells(&state, columnIdx);

	for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;
		global Segment* segments = getSegments(&state, columnIdx, i);

		for (int a = 0 ; a < cell->segmentCount; ++a)
		{
			global Segment* segment = segments + a;

			// Cache segment activity here..
			int activity = 0;
			int fullActivity = 0;
			int learnActivity = 0;
			int fullLearnActivity = 0;
			global Synapse* synapses = getSynapses(&state, columnIdx, i, a);
			for (int b = 0 ; b < segment->synapseCount; ++b)
			{
				global Synapse* syn = synapses + b;

				global Cell* targetCell = getCells(&state, syn->targetColumn) + syn->targetCell;

				syn->targetCellState |= targetCell->state & 0xF;

				if (getCellState(targetCell->state, NOW, ACTIVESTATE))
				{
					fullActivity++;
					if (syn->permanence > CONNECTED_PERMANENCE)
						activity++;
				}
				if (getCellState(targetCell->state, NOW, LEARNSTATE))
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
				setCellState(cell, PREDICTIVESTATE);

				getSegmentActiveSynapses(&state, columnIdx, i, a, NOW, false, seedValue);

				BestMatchingSegmentStruct bestMatch = getBestMatchingSegment(&state, columnIdx, i, WAS);
				getSegmentActiveSynapses(&state, columnIdx, i, bestMatch.segmentIdx, WAS, true, seedValue);
			}
		}
	}
}

void kernel updateSynapses(
	global Cell* g_cells,
	global Segment* g_segments,
	global Synapse* g_synapses,
	global char* resultBuffer)
{
	State state = makeState(g_cells, g_segments, g_synapses);
	int columnIdx = get_global_id(0);
	global char* result = resultBuffer + columnIdx;

	bool columnActive = false;

	global Cell* cells = getCells(&state, columnIdx);
	for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;
		if (getCellState(cell->state, NOW, LEARNSTATE))
		{
			adaptSegments(&state, columnIdx, i, true);
		}
		else if(!getCellState(cell->state, NOW, PREDICTIVESTATE) && getCellState(cell->state, WAS, PREDICTIVESTATE))
		{
			adaptSegments(&state, columnIdx, i, false);
		}
		if (getCellState(cell->state, NOW, ACTIVESTATE|PREDICTIVESTATE))
		{
			columnActive = true;
		}
	}
	*result = columnActive;
}
