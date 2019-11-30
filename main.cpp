#include <vulkan/vulkan.hpp>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <iostream>
#include <stdexcept>
#include <optional>

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

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
	VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(
	VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
	VkAllocationCallbacks const* pAllocator) {

	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;

	bool isComplete() {
		return graphicsFamily.has_value();
	}
};

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
	vk::UniqueDebugUtilsMessengerEXT debugMessenger;
	vk::PhysicalDevice physicalDevice;

	void initWindow() {
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			throw std::runtime_error("failed to initialise SDL!");

		window = SDL_CreateWindow("Vulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_VULKAN);
		if (!window)
			throw std::runtime_error("failed to create window!");
	}

	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		pickPhysicalDevice();
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

		vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo = getDebugUtilsMessengerCreateInfo();
		vk::InstanceCreateInfo instanceCreateInfo(
			{}, &applicationInfo,
			static_cast<uint32_t>(layerCount), validationLayers.data(),
			static_cast<uint32_t>(extensions.size()), extensions.data()
		);
		instanceCreateInfo.pNext = &debugMessengerCreateInfo;

		instance = vk::createInstanceUnique(
			instanceCreateInfo
		);
	}

	vk::DebugUtilsMessengerCreateInfoEXT getDebugUtilsMessengerCreateInfo() {
		return vk::DebugUtilsMessengerCreateInfoEXT(
			{},
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
			&debugCallback
		);
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers)
			return;

		debugMessenger = instance->createDebugUtilsMessengerEXTUnique(
			getDebugUtilsMessengerCreateInfo()
		);
	}

	void pickPhysicalDevice() {
		bool deviceFound = false;
		std::vector<vk::PhysicalDevice> physicalDevices = instance->enumeratePhysicalDevices();

		if (physicalDevices.size() == 0)
			throw std::runtime_error("failed to fin GPUs with Vulkan support!");

		for (const auto& device : physicalDevices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				deviceFound = true;
				break;
			}
		}

		if (!deviceFound)
			throw std::runtime_error("failed to find a suitable GPU!");
	}

	bool isDeviceSuitable(vk::PhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		return indices.isComplete();
	}

	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
		QueueFamilyIndices indices;

		std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
				indices.graphicsFamily = i;

			if (indices.isComplete())
				break;

			i++;
		}

		return indices;
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

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {

		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
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