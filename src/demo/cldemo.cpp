#include "../clregion.h"
#include "util.h"
#include "sensor.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include <thread>
#include <sstream>
#include <random>

inline void checkError(cl_int err, const char * name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

bool spaceDown = false;
void checkEvents(bool* quit)
{
	SDL_Event e;
	while (SDL_PollEvent(&e))
	{
		if (e.type == SDL_QUIT)
			*quit = true;
		if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_SPACE)
		{
			spaceDown = true;
		}
		if (e.type == SDL_KEYUP && e.key.keysym.sym == SDLK_SPACE)
		{
			spaceDown = false;
		}
// 		if (e.type == SDL_MOUSEBUTTONDOWN)
// 			*quit = true;
	}
}

void genData(int width, std::vector<cl_char>& data, double timer)
{
	static std::mt19937 gen;

	for (int a = 0; a < width; ++a)
	{
		for (int i = 0; i < width; ++i)
		{
			double x = double(i)/width - 0.5;
			double y = double(a)/width - 0.5;
			double p = pow(sin(x+y+timer*0.01), 20);

			bool k = std::bernoulli_distribution(p)(gen);

			data[i+width*a] = k;
		}
	}
}

void demo1Loop(SDL_Window* window)
{
	// init opencl context here
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

	int columnWidth = 32;
	int columns = columnWidth*columnWidth;
	int inputWidth = 32;
	int inputSize = inputWidth*inputWidth;

	int inhibitionRadius = 5;
	int receptiveFieldRadius = 5;
	CLRegion region(defaultDevice, context,
		CLTopology::localInhibition2D(inputWidth, inputWidth, columnWidth, columnWidth, inhibitionRadius, receptiveFieldRadius),
		CLArgs()
	);

	double timer = 0;

	int lastRPS = 0;
	int curRPS = 0;
	time_t lastTime = time(nullptr);

	glTranslatef(-0.5, -0.5, 0);

	std::vector<cl_char> dataIn(inputSize);
	std::vector<cl_char> dataOut(columns);
	while(!done)
	{
		curRPS ++;
		time_t timeNow = time(nullptr);
		if (timeNow != lastTime)
		{
			lastRPS = curRPS;
			curRPS = 0;
			lastTime = timeNow;

			CLStats stats = region.getStats();
			std::cout << lastRPS << " " << stats.averageBoost << " " << stats.averageDutyCycle << " " << stats.totalSegments << " " << stats.totalSynapses << std::endl;
		}

		checkEvents(&done);

		SDL_GL_SwapWindow(window);
		glClearColor ( 0.0, 0.0, 0.0, 1.0 );
		glClear ( GL_COLOR_BUFFER_BIT );
		timer += 1;

		// Generate some data

		genData(inputWidth, dataIn, timer);

		glColor3d(1,1,1);

		glBegin(GL_QUADS);
		for (int a = 0; a < inputWidth; ++a)
		{
			for (int i = 0; i < inputWidth; ++i)
			{
				double x = double(i)/(inputWidth-1) - 0.5;
				double y = double(a)/(inputWidth-1) - 0.5;
				double nx = double(i+1)/(inputWidth-1) - 0.5;
				double ny = double(a+1)/(inputWidth-1) - 0.5;

				bool k = dataIn[i+inputWidth*a];
				if (k)
				{
					glVertex2d(x, y);
					glVertex2d(nx, y);
					glVertex2d(nx, ny);
					glVertex2d(x, ny);
				}
			}
		}
		glEnd();
		region.write(dataIn, dataOut);

		glBegin(GL_QUADS);
		for (int a = 0; a < columnWidth; ++a)
		{
			for (int i = 0; i < columnWidth; ++i)
			{
				double x = double(i)/(columnWidth-1) - 0.5;
				double y = double(a)/(columnWidth-1) - 0.5;
				double nx = double(i+1)/(columnWidth-1) - 0.5;
				double ny = double(a+1)/(columnWidth-1) - 0.5;

				bool k = dataOut[i+columnWidth*a];
				if (k)
				{
					glVertex2d(1+x, y);
					glVertex2d(1+nx, y);
					glVertex2d(1+nx, ny);
					glVertex2d(1+x, ny);
				}
			}
		}
		glEnd();
		std::vector<double> cast;
		region.backwards(dataOut, cast);
		glBegin(GL_QUADS);
		for (int a = 0; a < inputWidth; ++a)
		{
			for (int i = 0; i < inputWidth; ++i)
			{
				double x = double(i)/(inputWidth-1) - 0.5;
				double y = double(a)/(inputWidth-1) + 0.5;
				double nx = double(i+1)/(inputWidth-1) - 0.5;
				double ny = double(a+1)/(inputWidth-1) + 0.5;

				double k = cast[i+inputWidth*a];
				k /= 5;
				if (k > 1) k = 1;
				glColor3d(k, k, k);

				glVertex2d(x, y);
				glVertex2d(nx, y);
				glVertex2d(nx, ny);
				glVertex2d(x, ny);
			}
		}
		glEnd();
	}
}

void demo2Loop(SDL_Window* window)
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

int main(int argc, char *argv[])
{
	if (SDL_Init(SDL_INIT_EVERYTHING) == -1)
	{
		std::cout << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Window* window = SDL_CreateWindow("cldemo", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
// 	SDL_Window* window = SDL_CreateWindow("cldemo", 200, 200, 800, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	if (window == nullptr)
	{
		std::cout << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (renderer == nullptr)
	{
		std::cout << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_GLContext glContext = SDL_GL_CreateContext(window);
	SDL_GL_SetSwapInterval(0);

	int selection = 0;
	if (argc>1)
	{
		std::stringstream str;
		str << argv[1];
		str >> selection;
	}

	while(std::cin)
	{
		if (selection == 0)
		{
			std::cout << "Pick demo: \n\t1) stochastic pattern sweep\n\t2) periodic function prediction\n> ";
			std::cin >> selection;
		}
		if (selection == 1)
		{
			demo1Loop(window);
			break;
		}
		else if(selection == 2)
		{
			demo2Loop(window);
			break;
		}
		selection = 0;
	}

	SDL_GL_DeleteContext(glContext);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
