#include "clregion.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"
#include <CL/cl.hpp>

#include <random>

inline void checkError(cl_int err, const char * name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void checkEvents(bool* quit)
{
	SDL_Event e;
	while (SDL_PollEvent(&e))
	{
		if (e.type == SDL_QUIT)
			*quit = true;
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

void mainLoop(SDL_Window* window)
{
	// init opencl context here
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);

	if (platformList.empty())
	{
		throw std::runtime_error("No OpencL platforms available");
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
	CLRegion region(defaultDevice, context, columns, inputSize);

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
	}
}

int main()
{
	if (SDL_Init(SDL_INIT_EVERYTHING) == -1)
	{
		std::cout << SDL_GetError() << std::endl;
		return 1;
	}

	//	SDL_Window* window = SDL_CreateWindow("cldemo", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 512, 512, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	SDL_Window* window = SDL_CreateWindow("cldemo", 200, 200, 800, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
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
	mainLoop(window);

	SDL_GL_DeleteContext(glContext);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
