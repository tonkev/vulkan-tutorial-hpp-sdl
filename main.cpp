#include <vulkan/vulkan.hpp>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <iostream>
#include <stdexcept>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	SDL_Window* window = nullptr;
	SDL_Event event;
	bool quitting = false;

	vk::UniqueInstance instance;

	void initWindow() {
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			throw std::runtime_error("failed to initialise SDL!");

		window = SDL_CreateWindow("Vulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_VULKAN);
		if (!window)
			throw std::runtime_error("failed to create window!");
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
		if (enableValidationLayers && !checkValidationLayerSupport())
			throw std::runtime_error("validation layers requested, but not available!");

		vk::ApplicationInfo applicationInfo(
			"Hello Triangle", VK_MAKE_VERSION(1, 0, 0),
			"No Engine", VK_MAKE_VERSION(1, 0, 0),
			VK_API_VERSION_1_0
		);

		std::vector<const char*> extensions = getRequiredExtensions();

		size_t layerCount = 0;
		if (enableValidationLayers)
			layerCount = validationLayers.size();

		instance = vk::createInstanceUnique(
			vk::InstanceCreateInfo(
				{},
				&applicationInfo,
				static_cast<uint32_t>(layerCount), validationLayers.data(),
				static_cast<uint32_t>(extensions.size()), extensions.data()
			)
		);
	}

	std::vector<const char*> getRequiredExtensions() {
		unsigned int extensionCount;
		if (!SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr))
			throw std::runtime_error("failed to get required SDL extension count!");

		std::vector<const char*> extensions(extensionCount);
		if (!SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensions.data()))
			throw std::runtime_error("failed to get required SDL extensions!");

		if (enableValidationLayers)
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		return extensions;
	}

	bool checkValidationLayerSupport() {
		std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName)) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
				return false;
		}

		return true;
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