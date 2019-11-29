#include <vulkan/vulkan.hpp>

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

	vk::UniqueInstance instance;

	void initWindow() {
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			throw std::runtime_error("Failed to initialise SDL");

		window = SDL_CreateWindow("Vulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_VULKAN);
		if (!window)
			throw std::runtime_error("Failed to create window");
	}

	void initVulkan() {
		createInstance();
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

	void createInstance() {
		vk::ApplicationInfo applicationInfo(
			"Hello Triangle", VK_MAKE_VERSION(1, 0, 0),
			"No Engine", VK_MAKE_VERSION(1, 0, 0),
			VK_API_VERSION_1_0
		);

		unsigned int count;
		if (!SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr))
			throw std::runtime_error("Failed to get required SDL extension count");

		std::vector<const char*> extensions(count);
		if (!SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data()))
			throw std::runtime_error("Failed to get required SDL extensions");

		instance = vk::createInstanceUnique(
			vk::InstanceCreateInfo(
				{},
				&applicationInfo,
				0, nullptr,
				count, extensions.data()
			)
		);
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