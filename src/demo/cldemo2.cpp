#include <thread>
#include "../clregion.h"
#include "util.h"
#include "sensor.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

void checkEvents(bool* quit);
void demo2Loop(SDL_Window* window, bool& spaceDown)
{
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);

	if (platformList.empty())
	{
		throw std::runtime_error("No OpenCL platforms available");
	}

	auto& defaultPlatform = platformList.front();
	std::vector< cl::Device > deviceList;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
	if (deviceList.empty())
	{
		throw std::runtime_error("OpenCL platform contains no devices");
	}

	auto& defaultDevice = deviceList.front();
	cl::Context context({defaultDevice});

	bool done = false;

	int columns = 32*32;
	int inputSize = 512;

	int inhibitionRadius = 5;
	int receptiveFieldRadius = 5;
	CLArgs args;

	args.ColumnProximalSynapseMinOverlap = 5;
	args.ColumnProximalSynapseCount = 40;

	CLRegion region(defaultDevice, context,
					CLTopology::line(inputSize, columns, inhibitionRadius, receptiveFieldRadius),
					args
	);

	// 	glTranslatef(-1, -1, 0);

	std::vector<cl_char> dataIn(inputSize);
	std::vector<cl_char> dataOut(columns);

	std::vector<double> noisyRemap(inputSize);

	double timer = 0;
	const double dt = 0.01;
	Sensor sensor(inputSize, 32);

	int iterCount = 0;

	const int temporalPoolerThreshold = 10000;

	const int windowSize = 100;
	std::vector<double> predictions(0.05 * windowSize/dt, 0);
	int windowPosition = 0;

	auto predict = [&](double input, bool /*learning*/)
	{
		// Encode to SDR via an instance of the Sensor class
		dataIn = sensor.encode(input);

		// Feed SDR to region, receive activation in dataOut
		region.write(dataIn, dataOut, iterCount++ > temporalPoolerThreshold);

		// Find out what kind of input would cause this kind of region activation (noisy backwards convolution)
		region.backwards(dataOut, noisyRemap);

		// Use sensor to find out approximate input value that would cause this kind of SDR
		return sensor.decode(noisyRemap);
	};

	while(!done)
	{
		timer += dt;
		checkEvents(&done);

		SDL_GL_SwapWindow(window);
		glClearColor ( 0.0, 0.0, 0.0, 1.0 );
		glClear ( GL_COLOR_BUFFER_BIT );

		glColor3d(1,0,0);
		glBegin(GL_LINE_STRIP);
		for(int i = 0; i < int(predictions.size()); ++i)
		{
			glVertex2d(-1+1.0-i/(predictions.size()-1.0), predictions[(i + windowPosition) % predictions.size()]);
		}
		glEnd();
		glColor3d(1,1,1);
		glBegin(GL_LINE_STRIP);
		for(int i = 0; i < windowSize; ++i)
		{
			glVertex2d(-1+i/(windowSize-1.0), 0.2*sin(i/20.0+timer));
		}
		glEnd();

		// Generate one double value
		double input = 0.2*sin((windowSize-1.0)/20.0+timer);
		double output = predict(input, true);

		// Store output history for rendering
		if (--windowPosition < 0)
			windowPosition = predictions.size()-1;

		predictions[windowPosition] = output;

		// Draw sensor-encoded data
		glBegin(GL_QUADS);
		for (int i = 0 ; i < inputSize; ++i)
		{
			double k = 2.0 / inputSize;
			glColor3f(1,1,1);
			if (dataIn[i])
			{
				glVertex2d(k*i-1, k-1);
				glVertex2d(k*i-1, k*2-1);
				glVertex2d(k*(i+1)-1, k*2-1);
				glVertex2d(k*(i+1)-1, k-1);
			}

			double s = noisyRemap[i];
			s /= 5;
			if (s > 1) s = 1.0;
			glColor3f(s,s,s);

			{
				glVertex2d(k*i-1, -1);
				glVertex2d(k*i-1, k-1);
				glVertex2d(k*(i+1)-1, k-1);
				glVertex2d(k*(i+1)-1, -1);
			}
		}
		glColor3f(1,1,1);
		// Draw region activation
		for (int i = 0 ; i < columns; ++i)
		{
			double k = 2.0 / columns;
			if (dataOut[i])
			{
				glVertex2d(k*i-1, 1);
				glVertex2d(k*i-1, 1-k);
				glVertex2d(k*(i+1)-1, 1-k);
				glVertex2d(k*(i+1)-1, 1);
			}
		}

		glEnd();

		// Draw resulting line
		glBegin(GL_LINES);
		glVertex2d(0, output);
		glVertex2d(1, output);
		glEnd();

		if (!spaceDown)
			std::this_thread::sleep_for(std::chrono::milliseconds(1000/240));

		//CLStats stats = region.getStats();
		//std::cout << stats.averageBoost << " " << stats.averageDutyCycle << " " << stats.totalSegments << " " << stats.totalSynapses << std::endl;
	}
}
