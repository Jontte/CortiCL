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

	float activeDutyCycle; // how often this segment is active when the cell is active
} Segment;

typedef struct
{
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
	columnIdx * CELL_SEGMENT_COUNT * COLUMN_CELL_COUNT
	+ cellIdx * CELL_SEGMENT_COUNT];
}
inline global Synapse* getSynapses(const State* state, int columnIdx, int cellIdx, int segmentIdx)
{
	return &state->synapses[
	columnIdx * SEGMENT_SYNAPSE_COUNT * CELL_SEGMENT_COUNT * COLUMN_CELL_COUNT
	+ cellIdx * SEGMENT_SYNAPSE_COUNT * CELL_SEGMENT_COUNT
	+ segmentIdx * SEGMENT_SYNAPSE_COUNT
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

uint random(uint2* seedValue)
{
	uint seed = (seedValue->x++) + get_global_id(0);
	uint t = seed ^ (seed << 11);
	return seedValue->y ^ (seedValue->y >> 19) ^ (t ^ (t >> 8));
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
void resetSynapse(const State* state, global Synapse* synapse, bool connectToLearningCell, TimeStep learningCellWhen, uint2* randomState)
{
	// If we fail to connect to a learning cell, fallback to a randomly selected cell
	if (connectToLearningCell)
	{
		int randOffset = random(randomState) % get_global_size(0);

		int targetColumn = -1;
		int targetCell = -1;
		int columnCount = get_global_size(0);
		for (int i = 0; i < columnCount; ++i)
		{
			int testColumnId = (randOffset+i) % get_global_size(0);

			global Cell* testColumn = getCells(state, testColumnId);

			if (testColumnId == get_global_id(0)) // Skip self...
				continue;

			for (int a = 0; a < COLUMN_CELL_COUNT; ++a)
			{
				global Cell* testCell = testColumn + a;
				if (getCellState(testCell->state, learningCellWhen, LEARNSTATE))
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

		// Check if we found a cell
		if (targetColumn != -1)
		{
			synapse->targetColumn = targetColumn;
			synapse->targetCell = targetCell;
			synapse->permanence = CONNECTED_PERMANENCE*2;
			synapse->targetCellState = 0;
			return;
		}
	}

	// Pick random column, skip self
	int targetColumn = random(randomState) % (get_global_size(0)-1);
	if (targetColumn >= get_global_id(0))
		targetColumn++;
	// Pick random cell
	int targetCell = random(randomState) % COLUMN_CELL_COUNT;

	synapse->targetColumn = targetColumn;
	synapse->targetCell = targetCell;
	synapse->permanence = CONNECTED_PERMANENCE*2;
	synapse->targetCellState = 0;
}

global Segment* getActiveSegment(const State* state, int columnIdx, int cellIdx, TimeStep when, CellState cellState)
{
	global Cell* cell = getCells(state, columnIdx) + cellIdx;
	global Segment* segments = getSegments(state, columnIdx, cellIdx);

	bool activeSequenceSegments = false;
	for (int i = 0; i < CELL_SEGMENT_COUNT; ++i)
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

	for (int i = 0; i < CELL_SEGMENT_COUNT; ++i)
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
	for (int i = 0; i < CELL_SEGMENT_COUNT; ++i)
	{
		global Segment* seg = segments + i;

		// Check synapses that are not even fully connected
		int activity = seg->fullActivity[ACTIVESTATE][when];

		if (activity > bestActivity || i == 0)
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

	int bestActivityIdx = -1;
	int bestActivity = 0;

	// Return cell and segment with highest activity
	for (int i = 0; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;
		BestMatchingSegmentStruct bestSegment =
			getBestMatchingSegment(state, columnIdx, i, when);

		if (bestSegment.activity > bestActivity || i == 0)
		{
			bestActivity = bestSegment.activity;
			ret.cell = cell;
			ret.cellIdx = i;
			ret.segment = bestSegment.segment;
			ret.segmentIdx = bestSegment.segmentIdx;
		}
	}
	return ret;
}

void getSegmentActiveSynapses(
	const State* state,
	int columnIdx,
	int cellIdx,
	int segmentIdx,
	TimeStep when,
	bool newSynapses,
	uint2* randomState
	)
{
	global Segment* segment = getSegments(state, columnIdx, cellIdx) + segmentIdx;
	global Synapse* synapses = getSynapses(state, columnIdx, cellIdx, segmentIdx);

	// If no changes have been queued, set permamenceQueued of each synapse to match current permanence
	bool hasChanges = segment->hasQueuedChanges;
	if (!hasChanges)
	{
		for (int i = 0 ; i < SEGMENT_SYNAPSE_COUNT; ++i)
		{
			global Synapse* synapse = synapses + i;
			synapse->permanenceQueued = synapse->permanence;
		}
		segment->sequenceSegmentQueued = segment->sequenceSegment;
		segment->hasQueuedChanges = true;
	}

	// Find active synapses in segment
	for (int i = 0 ; i < SEGMENT_SYNAPSE_COUNT; ++i)
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

	// Enhance segment by connecting some of the worst synapses to learning cells
	if (newSynapses)
	{
		// For each bad synapse...
		for (int b = 0; b < SEGMENT_SYNAPSE_COUNT; ++b)
		{
			global Synapse* synapse = synapses + b;

			if (synapse->permanence > CONNECTED_PERMANENCE / 2.0f)
				continue;

			resetSynapse(state, synapse, true, when, randomState);
		}
	}
}

void adaptSegments(const State* state, int columnIdx, int cellIdx, bool positiveReinforcement)
{
	global Cell* cell = getCells(state, columnIdx) + cellIdx;
	global Segment* segments = getSegments(state, columnIdx, cellIdx);

	for (int i = 0; i < CELL_SEGMENT_COUNT; ++i)
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
			for (int a = 0; a < SEGMENT_SYNAPSE_COUNT; ++a)
			{
				global Synapse* synapse = synapses + a;
				synapse->permanence = synapse->permanenceQueued;
			}
		}
		else
		{
			// Oops, misprediction. Synapses that had their permanence enhanced are flipped, the rest stay untouched
			for (int a = 0; a < SEGMENT_SYNAPSE_COUNT; ++a)
			{
				global Synapse* synapse = synapses + a;
				if (synapse->permanenceQueued > synapse->permanence)
				{
					synapse->permanence += synapse->permanence - synapse->permanenceQueued;
				}
			}
		}
		for (int a = 0; a < SEGMENT_SYNAPSE_COUNT; ++a)
		{
			global Synapse* synapse = synapses + a;
			if (synapse->permanence > 1.0f)
				synapse->permanence = 1.0f;
			else if (synapse->permanence < 0.0f)
				synapse->permanence = 0.0f;
		}
	}
}



void kernel initRegion(
	global Cell* g_cells,
	global Segment* g_segments,
	global Synapse* g_synapses,
	uint2 randomState)
{
	State state = makeState(g_cells, g_segments, g_synapses);
	int columnIdx = get_global_id(0);

	// Get cells of the current column
	global Cell* cells = getCells(&state, columnIdx);

	for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;
		cell->state = 0;

		global Segment* segments = getSegments(&state, columnIdx, i);
		for (int a = 0; a < CELL_SEGMENT_COUNT; ++a)
		{
			global Segment* segment = segments + a;
			segment->activity[0][WAS] = 0;
			segment->activity[1][WAS] = 0;
			segment->activity[0][NOW] = 0;
			segment->activity[1][NOW] = 0;
			segment->fullActivity[0][WAS] = 0;
			segment->fullActivity[1][WAS] = 0;
			segment->fullActivity[0][NOW] = 0;
			segment->fullActivity[1][NOW] = 0;
			segment->sequenceSegment = false;
			segment->sequenceSegmentQueued = false;
			segment->hasQueuedChanges = false;
			segment->activeDutyCycle = 0;

			global Synapse* synapses = getSynapses(&state, columnIdx, i, a);
			for (int b = 0; b < SEGMENT_SYNAPSE_COUNT; ++b)
			{
				resetSynapse(&state, synapses+b, false, NOW, &randomState);
			}
		}
	}
}

// Reset segment with the worst duty cycle
void kernel refineRegion(
	global Cell* g_cells,
	global Segment* g_segments,
	global Synapse* g_synapses,
	uint2 randomState)
{
	State state = makeState(g_cells, g_segments, g_synapses);
	int columnIdx = get_global_id(0);

	// Get cells of the current column
	global Cell* cells = getCells(&state, columnIdx);

	for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;

		float worstDutyCycle = 0;
		int worstDutyCycleIndex = -1;

		global Segment* segments = getSegments(&state, columnIdx, i);
		for (int a = 0; a < CELL_SEGMENT_COUNT; ++a)
		{
			global Segment* segment = segments+a;
			if (segment->activeDutyCycle < worstDutyCycle || a == 0)
			{
				worstDutyCycle = segment->activeDutyCycle;
				worstDutyCycleIndex = a;
			}
		}

		global Synapse* synapses = getSynapses(&state, columnIdx, i, worstDutyCycleIndex);
		for (int a = 0; a < SEGMENT_SYNAPSE_COUNT; ++a)
		{
			resetSynapse(&state, synapses+a, true, NOW, &randomState);
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
		cell->state = (cell->state << 4) & 0xF0;

		global Segment* segments = getSegments(&state, columnIdx, i);
		for (int a = 0; a < CELL_SEGMENT_COUNT; ++a)
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
			for (int b = 0; b < SEGMENT_SYNAPSE_COUNT; ++b)
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
	uint2 randomState)
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

		getSegmentActiveSynapses(&state, columnIdx, learnCellIdx, learnSegmentIdx, WAS, true, &randomState);
		learnSegment->sequenceSegmentQueued = true;
	}
}

void kernel computePredictiveState(
	global Cell* g_cells,
	global Segment* g_segments,
	global Synapse* g_synapses,
	global const char* activeColumns,
	uint2 randomState)
{
	State state = makeState(g_cells, g_segments, g_synapses);

	int columnIdx = get_global_id(0);
	global Cell* cells = getCells(&state, columnIdx);

	for (int i = 0 ; i < COLUMN_CELL_COUNT; ++i)
	{
		global Cell* cell = cells + i;
		global Segment* segments = getSegments(&state, columnIdx, i);

		for (int a = 0 ; a < CELL_SEGMENT_COUNT; ++a)
		{
			global Segment* segment = segments + a;

			// Cache segment activity here..
			int activity = 0;
			int fullActivity = 0;
			int learnActivity = 0;
			int fullLearnActivity = 0;
			global Synapse* synapses = getSynapses(&state, columnIdx, i, a);
			for (int b = 0 ; b < SEGMENT_SYNAPSE_COUNT; ++b)
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

//			if (segmentActive(segment, NOW, ACTIVESTATE))
//			if (segmentActivity(segment, NOW, ACTIVESTATE) > SEGMENT_ACTIVATION_THRESHOLD)
			if (segment->activity[0][NOW] > SEGMENT_ACTIVATION_THRESHOLD)
			{
				setCellState(cell, PREDICTIVESTATE);

				getSegmentActiveSynapses(&state, columnIdx, i, a, NOW, false, &randomState);

				BestMatchingSegmentStruct bestMatch = getBestMatchingSegment(&state, columnIdx, i, WAS);
				getSegmentActiveSynapses(&state, columnIdx, i, bestMatch.segmentIdx, WAS, true, &randomState);
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

		// Update segment duty cycles
		if (getCellState(cell->state, NOW, ACTIVESTATE))
		{
			global Segment* segments = getSegments(&state, columnIdx, i);
			for (int a = 0; a < CELL_SEGMENT_COUNT; ++a)
			{
				global Segment* segment = segments+a;
				bool active = segmentActive(segment, NOW, ACTIVESTATE);

				const float persistence = 0.95f;
				segment->activeDutyCycle *= persistence;
				segment->activeDutyCycle += active * (1.0f - persistence);
			}
		}
		if (getCellState(cell->state, NOW, PREDICTIVESTATE))
		{
			columnActive = true;
		}
	}
	*result = columnActive;
}
