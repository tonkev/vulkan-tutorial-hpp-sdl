#include <vulkan/vulkan.hpp>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <optional>
#include <set>

const int WIDTH = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapchainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
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
	vk::UniqueSurfaceKHR surface;

	vk::PhysicalDevice physicalDevice;
	vk::UniqueDevice device;

	vk::Queue graphicsQueue;
	vk::Queue presentQueue;

	vk::UniqueSwapchainKHR swapchain;
	std::vector<vk::Image> swapchainImages;
	vk::Format swapchainImageFormat;
	vk::Extent2D swapchainExtent;
	std::vector<vk::UniqueImageView> swapchainImageViews;
	std::vector<vk::UniqueFramebuffer> swapchainFramebuffers;

	vk::UniqueRenderPass renderPass;
	vk::UniquePipelineLayout pipelineLayout;
	std::vector<vk::UniquePipeline> graphicsPipelines;

	vk::UniqueCommandPool commandPool;
	std::vector<vk::UniqueCommandBuffer> commandBuffers;

	std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
	std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
	std::vector<vk::UniqueFence> inFlightFences;
	std::vector<size_t> imagesInFlight;
	size_t currentFrame = 0;

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
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapchain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createCommandBuffers();
		createSyncObjects();
	}

	void mainLoop() {
		while (!quitting) {
			while (SDL_PollEvent(&event) != 0) {
				if (event.type == SDL_QUIT) {
					quitting = true;
				}
			}
			drawFrame();
		}

		device->waitIdle();
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

	void createSurface() {
		vk::SurfaceKHR tmpSurface;
		if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(instance.get()), reinterpret_cast<VkSurfaceKHR*>(&tmpSurface)))
			throw std::runtime_error("failed to create SDL surface!");
		surface = vk::UniqueSurfaceKHR(tmpSurface, instance.get());
	}

	void pickPhysicalDevice() {
		bool deviceFound = false;
		std::vector<vk::PhysicalDevice> physicalDevices = instance->enumeratePhysicalDevices();

		if (physicalDevices.size() == 0)
			throw std::runtime_error("failed to find GPUs with Vulkan support!");

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

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			queueCreateInfos.push_back(
				vk::DeviceQueueCreateInfo({}, queueFamily, 1, &queuePriority)
			);
		}
		vk::PhysicalDeviceFeatures deviceFeatures = {};

		uint32_t enabledLayerCount = 0;
		if (enableValidationLayers)
			enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		
		device = physicalDevice.createDeviceUnique(
			vk::DeviceCreateInfo(
				{},
				static_cast<uint32_t>(queueCreateInfos.size()), queueCreateInfos.data(),
				enabledLayerCount, validationLayers.data(),
				static_cast<uint32_t>(deviceExtensions.size()), deviceExtensions.data(),
				&deviceFeatures
			)
		);

		graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
		presentQueue = device->getQueue(indices.presentFamily.value(), 0);
	}

	void createSwapchain() {
		SwapchainSupportDetails swapchainSupport = querySwapchainSupport(physicalDevice);

		vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
		vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
		vk::Extent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

		uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
		if (swapchainSupport.capabilities.maxImageCount > 0 && imageCount > swapchainSupport.capabilities.maxImageCount)
			imageCount = swapchainSupport.capabilities.maxImageCount;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

		vk::SharingMode sharingMode = vk::SharingMode::eExclusive;
		if (indices.graphicsFamily != indices.presentFamily)
			sharingMode = vk::SharingMode::eConcurrent;
		
		swapchain = device->createSwapchainKHRUnique(
			vk::SwapchainCreateInfoKHR(
				vk::SwapchainCreateFlagsKHR(), surface.get(),
				imageCount, surfaceFormat.format,
				surfaceFormat.colorSpace, extent,
				1, vk::ImageUsageFlagBits::eColorAttachment,
				sharingMode, 2, queueFamilyIndices,
				swapchainSupport.capabilities.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque,
				presentMode, true,
				nullptr
			)
		);

		swapchainImages = device->getSwapchainImagesKHR(swapchain.get());
		swapchainImageFormat = surfaceFormat.format;
		swapchainExtent = extent;
	}

	void createImageViews() {
		swapchainImageViews.resize(swapchainImages.size());

		for (size_t i = 0; i < swapchainImages.size(); ++i) {
			swapchainImageViews[i] = device->createImageViewUnique(
				vk::ImageViewCreateInfo(
					vk::ImageViewCreateFlags(), swapchainImages[i],
					vk::ImageViewType::e2D, swapchainImageFormat,
					vk::ComponentMapping(),
					vk::ImageSubresourceRange(
						vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1
					)
				)
			);
		}
	}

	void createRenderPass() {
		vk::AttachmentDescription colorAttachment(
			vk::AttachmentDescriptionFlags(),
			swapchainImageFormat, vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR
		);

		vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

		vk::SubpassDescription subpass;
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		vk::SubpassDependency dependency(
			(uint32_t) VK_SUBPASS_EXTERNAL, (uint32_t) 0,
			vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::AccessFlags(), vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
			vk::DependencyFlags()
		);

		renderPass = device->createRenderPassUnique(
			vk::RenderPassCreateInfo(
				vk::RenderPassCreateFlags(), 1, &colorAttachment, 1, &subpass, 1, &dependency
			)
		);
	}

	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("shaders/vert.spv");
		auto fragShaderCode = readFile("shaders/frag.spv");

		vk::UniqueShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		vk::UniqueShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo(
			vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, vertShaderModule.get(), "main"
		);
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo(
			vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fragShaderModule.get(), "main"
		);

		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
			vk::PipelineVertexInputStateCreateFlags(), 0, nullptr, 0, nullptr
		);

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
			vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList, false
		);

		vk::Viewport viewport(0.0f, 0.0f, (float) swapchainExtent.width, (float) swapchainExtent.height, 0.0f, 1.0f);

		vk::Rect2D scissor({0, 0}, swapchainExtent);

		vk::PipelineViewportStateCreateInfo viewportState(
			vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissor
		);

		vk::PipelineRasterizationStateCreateInfo rasterizer(
			vk::PipelineRasterizationStateCreateFlags(), false, false,
			vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise, 
			false, 0.0f, 0.0f, 0.0f, 1.0f
		);

		vk::PipelineMultisampleStateCreateInfo multisampling(
			vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1, 
			false, 1.0f, nullptr, false, false
		);

		vk::PipelineColorBlendAttachmentState colorBlendAttachment;
		colorBlendAttachment.blendEnable = false;
		colorBlendAttachment.colorWriteMask =
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo colorBlending(
			vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eClear, 1, &colorBlendAttachment
		);

		pipelineLayout = device->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 0, nullptr, 0, nullptr)
		);

		vk::GraphicsPipelineCreateInfo pipelineInfo;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.layout = pipelineLayout.get();
		pipelineInfo.renderPass = renderPass.get();
		pipelineInfo.subpass = 0;

		graphicsPipelines = device->createGraphicsPipelinesUnique({}, pipelineInfo);
	}

	void createFramebuffers() {
		swapchainFramebuffers.resize(swapchainImageViews.size());

		for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
			vk::ImageView attachments[] = {swapchainImageViews[i].get()};

			swapchainFramebuffers[i] = device->createFramebufferUnique(
				vk::FramebufferCreateInfo(
					vk::FramebufferCreateFlags(), renderPass.get(),
					1, attachments,
					swapchainExtent.width, swapchainExtent.height,
					1
				)
			);
		}
	}

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		commandPool = device->createCommandPoolUnique(
			vk::CommandPoolCreateInfo(
				vk::CommandPoolCreateFlags(), queueFamilyIndices.graphicsFamily.value()
			)
		);
	}

	void createCommandBuffers() {
		commandBuffers.resize(swapchainFramebuffers.size());

		commandBuffers = device->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo(
				commandPool.get(), vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(commandBuffers.size())
			)
		);

		for (size_t i = 0; i < commandBuffers.size(); ++i) {
			commandBuffers[i]->begin(
				vk::CommandBufferBeginInfo(
					vk::CommandBufferUsageFlags(), nullptr
				)
			);

			vk::ClearValue clearValue;
			clearValue.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
			vk::RenderPassBeginInfo renderPassBeginInfo(
				renderPass.get(), swapchainFramebuffers[i].get(), vk::Rect2D({0, 0}, swapchainExtent), (uint32_t) 1, &clearValue
			);
			commandBuffers[i]->beginRenderPass(&renderPassBeginInfo, vk::SubpassContents::eInline);

			commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipelines[0].get());

			commandBuffers[i]->draw(3, 1, 0, 0);

			commandBuffers[i]->endRenderPass();

			commandBuffers[i]->end();
		}
	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		imagesInFlight.resize(swapchainImages.size(), -1);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			imageAvailableSemaphores[i] = device->createSemaphoreUnique(vk::SemaphoreCreateInfo(vk::SemaphoreCreateFlags()));
			renderFinishedSemaphores[i] = device->createSemaphoreUnique(vk::SemaphoreCreateInfo(vk::SemaphoreCreateFlags()));
			inFlightFences[i] = device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
		}
	}

	void drawFrame() {
		device->waitForFences(1, &inFlightFences[currentFrame].get(), true, UINT64_MAX);

		vk::ResultValue<uint32_t> result = device->acquireNextImageKHR(swapchain.get(), (uint64_t)UINT64_MAX, imageAvailableSemaphores[currentFrame].get(), nullptr);
		
		if (imagesInFlight[result.value] != -1) 
			device->waitForFences(1, &inFlightFences[imagesInFlight[result.value]].get(), true, UINT64_MAX);
		imagesInFlight[result.value] = currentFrame;
		
		vk::PipelineStageFlags waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

		device->resetFences(1, &inFlightFences[currentFrame].get());

		graphicsQueue.submit(
			vk::SubmitInfo(
				(uint32_t) 1, &imageAvailableSemaphores[currentFrame].get(), &waitStages,
				(uint32_t) 1, &(commandBuffers[result.value].get()),
				(uint32_t) 1, &renderFinishedSemaphores[currentFrame].get()
			), inFlightFences[currentFrame].get()
		);

		presentQueue.presentKHR(
			vk::PresentInfoKHR(
				1, &renderFinishedSemaphores[currentFrame].get(),
				1, &swapchain.get(),
				&result.value, nullptr
			)
		);

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	vk::UniqueShaderModule createShaderModule(const std::vector<char>& code) {
		return device->createShaderModuleUnique(
			vk::ShaderModuleCreateInfo(
				vk::ShaderModuleCreateFlags(), code.size(), reinterpret_cast<const uint32_t*>(code.data())
			)
		);
	}

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
				return availableFormat;
		}

		return availableFormats[0];
	}

	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == vk::PresentModeKHR::eMailbox)
				return availablePresentMode;
		}

		return vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		}
		else {
			vk::Extent2D actualExtent = {WIDTH, HEIGHT};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	SwapchainSupportDetails querySwapchainSupport(vk::PhysicalDevice device) {
		SwapchainSupportDetails details;

		details.capabilities = device.getSurfaceCapabilitiesKHR(surface.get());
		details.formats = device.getSurfaceFormatsKHR(surface.get());
		details.presentModes = device.getSurfacePresentModesKHR(surface.get());

		return details;
	}

	bool isDeviceSuitable(vk::PhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool swapchainAdequate = false;
		if (checkDeviceExtensionSupport(device)) {
			SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device);
			swapchainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
		}

		return indices.isComplete() && swapchainAdequate;
	}

	bool checkDeviceExtensionSupport(vk::PhysicalDevice device) {
		std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
		QueueFamilyIndices indices;

		std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
				indices.graphicsFamily = i;

			if (device.getSurfaceSupportKHR(i, surface.get()))
				indices.presentFamily = i;

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

	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("failed to open file!");

		size_t fileSize = (size_t) file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
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