#include <SDL.h>
#include <SDL_vulkan.h>

#include <iostream>
#include <stdexcept>

const int WIDTH = 800;
const int HEIGHT = 600;

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	SDL_Window* window;
	SDL_Event event;
	bool quitting = false;

	void initWindow() {
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			throw std::runtime_error("Failed to initialise SDL");

		window = SDL_CreateWindow("Vulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
		if (!window)
			throw std::runtime_error("Failed to create window");
	}

	void initVulkan() {

	}

	void mainLoop() {
		while (!quitting) {
			while (SDL_PollEvent(&event) != 0) {
				if (event.type == SDL_QUIT) {
					quitting = true;
				}
			}
		}
	}

	void cleanup() {
		SDL_DestroyWindow(window);

		SDL_Quit();
	}
};

int main(int argc, char* argv[]) {
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}