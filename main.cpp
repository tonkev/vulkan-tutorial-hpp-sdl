#include <vulkan/vulkan.hpp>

#include <SDL.h>
#include <SDL_vulkan.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <chrono>
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

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription() {
		vk::VertexInputBindingDescription bindingDescription = {};

		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = vk::VertexInputRate::eVertex;

		return bindingDescription;
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
		std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
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
	vk::UniqueDescriptorSetLayout descriptorSetLayout;
	vk::UniquePipelineLayout pipelineLayout;
	std::vector<vk::UniquePipeline> graphicsPipelines;

	vk::UniqueCommandPool commandPool;

	vk::UniqueImage textureImage;
	vk::UniqueDeviceMemory textureImageMemory;
	vk::UniqueImageView textureImageView;
	vk::UniqueSampler textureSampler;

	vk::UniqueBuffer vertexBuffer;
	vk::UniqueDeviceMemory vertexBufferMemory;
	vk::UniqueBuffer indexBuffer;
	vk::UniqueDeviceMemory indexBufferMemory;

	std::vector<vk::UniqueBuffer> uniformBuffers;
	std::vector<vk::UniqueDeviceMemory> uniformBuffersMemory;

	vk::UniqueDescriptorPool descriptorPool;
	std::vector<vk::DescriptorSet> descriptorSets;

	std::vector<vk::UniqueCommandBuffer> commandBuffers;

	std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
	std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
	std::vector<vk::UniqueFence> inFlightFences;
	std::vector<size_t> imagesInFlight;
	size_t currentFrame = 0;

	void initWindow() {
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			throw std::runtime_error("failed to initialise SDL!");

		window = SDL_CreateWindow("Vulkan", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
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
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	void mainLoop() {
		while (!quitting) {
			while (SDL_PollEvent(&event) != 0) {
				if (event.type == SDL_QUIT) {
					quitting = true;
				}
				if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
					recreateSwapchain();
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
		deviceFeatures.samplerAnisotropy = true;

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
			swapchainImageViews[i] = createImageView(swapchainImages[i], swapchainImageFormat);
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

	void createDescriptorSetLayout() {
		vk::DescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
		uboLayoutBinding.pImmutableSamplers = nullptr;

		descriptorSetLayout = device->createDescriptorSetLayoutUnique(
			vk::DescriptorSetLayoutCreateInfo(
				vk::DescriptorSetLayoutCreateFlags(), 1, &uboLayoutBinding
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

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
			vk::PipelineVertexInputStateCreateFlags(), 
			1, &bindingDescription,
			2, attributeDescriptions.data()
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
			vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, 
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
			vk::PipelineLayoutCreateInfo(
				vk::PipelineLayoutCreateFlags(),
				1, &descriptorSetLayout.get(),
				0, nullptr
			)
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

	void createTextureImage() {
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		vk::DeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels)
			throw std::runtime_error("failed to load texture image!");

		vk::UniqueBuffer stagingBuffer;
		vk::UniqueDeviceMemory stagingBufferMemory;
		createBuffer(
			imageSize, vk::BufferUsageFlagBits::eTransferSrc, 
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory
		);

		void* data = device->mapMemory(stagingBufferMemory.get(), 0, imageSize, vk::MemoryMapFlags());
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		device->unmapMemory(stagingBufferMemory.get());

		stbi_image_free(pixels);

		createImage(
			texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal,
			textureImage, textureImageMemory
		);

		transitionImageLayout(textureImage.get(), vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		copyBufferToImage(stagingBuffer.get(), textureImage.get(), texWidth, texHeight);
		transitionImageLayout(textureImage.get(), vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
	}

	void createTextureImageView() {
		textureImageView = createImageView(textureImage.get(), vk::Format::eR8G8B8A8Unorm);
	}

	void createTextureSampler() {
		vk::SamplerCreateInfo samplerInfo = {};
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
		samplerInfo.anisotropyEnable = true;
		samplerInfo.maxAnisotropy = 16;
		samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
		samplerInfo.unnormalizedCoordinates = false;
		samplerInfo.compareEnable = false;
		samplerInfo.compareOp = vk::CompareOp::eAlways;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		textureSampler = device->createSamplerUnique(samplerInfo);
	}

	vk::UniqueImageView createImageView(vk::Image image, vk::Format format) {
		vk::ImageViewCreateInfo viewInfo = {};
		viewInfo.image = image;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		return device->createImageViewUnique(viewInfo);
	}

	void createImage(
			uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
			vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueImage& image, vk::UniqueDeviceMemory& imageMemory) {

		vk::ImageCreateInfo imageInfo = {};
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = vk::ImageLayout::eUndefined;
		imageInfo.usage = usage;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;
		imageInfo.samples = vk::SampleCountFlagBits::e1;

		image = device->createImageUnique(imageInfo);

		vk::MemoryRequirements memRequirements = device->getImageMemoryRequirements(image.get());

		vk::MemoryAllocateInfo allocInfo = {};
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		imageMemory = device->allocateMemoryUnique(allocInfo);

		device->bindImageMemory(image.get(), imageMemory.get(), 0);
	}

	void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
		std::vector<vk::UniqueCommandBuffer> commandBuffers = beginSingleTimeCommands();

		vk::ImageMemoryBarrier barrier = {};
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		vk::PipelineStageFlags sourceStage;
		vk::PipelineStageFlags destinationStage;

		if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;
		} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		} else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		commandBuffers[0]->pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(), nullptr, nullptr, barrier);

		endSingleTimeCommands(commandBuffers);
	}

	void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
		std::vector<vk::UniqueCommandBuffer> commandBuffers = beginSingleTimeCommands();

		vk::BufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {width, height, 1};

		commandBuffers[0]->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

		endSingleTimeCommands(commandBuffers);
	}

	void createVertexBuffer() {
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		vk::UniqueBuffer stagingBuffer;
		vk::UniqueDeviceMemory stagingBufferMemory;
		createBuffer(
			bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory
		);

		void* data = device->mapMemory(stagingBufferMemory.get(), 0, bufferSize, vk::MemoryMapFlags());
		memcpy(data, vertices.data(), (size_t)bufferSize);
		device->unmapMemory(stagingBufferMemory.get());

		createBuffer(
			bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vertexBuffer, vertexBufferMemory
		);

		copyBuffer(stagingBuffer.get(), vertexBuffer.get(), bufferSize);
	}

	void createIndexBuffer() {
		vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		vk::UniqueBuffer stagingBuffer;
		vk::UniqueDeviceMemory stagingBufferMemory;
		createBuffer(
			bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory
		);

		void* data = device->mapMemory(stagingBufferMemory.get(), 0, bufferSize, vk::MemoryMapFlags());
		memcpy(data, indices.data(), (size_t)bufferSize);
		device->unmapMemory(stagingBufferMemory.get());

		createBuffer(
			bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			indexBuffer, indexBufferMemory
		);

		copyBuffer(stagingBuffer.get(), indexBuffer.get(), bufferSize);
	}

	void createUniformBuffers() {
		vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(swapchainImages.size());
		uniformBuffersMemory.resize(swapchainImages.size());

		for (size_t i = 0; i < swapchainImages.size(); ++i) {
			createBuffer(
				bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				uniformBuffers[i], uniformBuffersMemory[i]
			);
		}
	}

	void createDescriptorPool() {
		vk::DescriptorPoolSize poolSize = {};
		poolSize.type = vk::DescriptorType::eUniformBuffer;
		poolSize.descriptorCount = static_cast<uint32_t>(swapchainImages.size());

		vk::DescriptorPoolCreateInfo poolInfo = {};
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = static_cast<uint32_t>(swapchainImages.size());

		descriptorPool = device->createDescriptorPoolUnique(poolInfo);
	}

	void createDescriptorSets() {
		std::vector<vk::DescriptorSetLayout> layouts(swapchainImages.size(), descriptorSetLayout.get());

		vk::DescriptorSetAllocateInfo allocInfo = {};
		allocInfo.descriptorPool = descriptorPool.get();
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapchainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets = device->allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < swapchainImages.size(); ++i) {
			vk::DescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers[i].get();
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			vk::WriteDescriptorSet descriptorWrite = {};
			descriptorWrite.dstSet = descriptorSets[i];
			descriptorWrite.dstBinding = 0;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pBufferInfo = &bufferInfo;

			device->updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
		}
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& bufferMemory) {
		vk::BufferCreateInfo bufferCreateInfo(
			vk::BufferCreateFlags(), size,
			usage, vk::SharingMode::eExclusive
		);

		buffer = device->createBufferUnique(bufferCreateInfo);

		vk::MemoryRequirements memRequirements = device->getBufferMemoryRequirements(buffer.get());

		bufferMemory = device->allocateMemoryUnique(
			vk::MemoryAllocateInfo(
				memRequirements.size,
				findMemoryType(memRequirements.memoryTypeBits, properties)
			)
		);

		device->bindBufferMemory(buffer.get(), bufferMemory.get(), 0);
	}

	std::vector<vk::UniqueCommandBuffer> beginSingleTimeCommands() {
		std::vector<vk::UniqueCommandBuffer> commandBuffers = device->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo(
				commandPool.get(), vk::CommandBufferLevel::ePrimary, 1
			)
		);

		commandBuffers[0]->begin(
			vk::CommandBufferBeginInfo(
				vk::CommandBufferUsageFlagBits::eOneTimeSubmit, nullptr
			)
		);

		return commandBuffers;
	}

	void endSingleTimeCommands(std::vector<vk::UniqueCommandBuffer>& commandBuffers) {
		commandBuffers[0]->end();

		vk::SubmitInfo submitInfo = {};
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[0].get();

		graphicsQueue.submit(submitInfo, nullptr);
		graphicsQueue.waitIdle();
	}

	void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
		std::vector<vk::UniqueCommandBuffer> commandBuffers = beginSingleTimeCommands();

		vk::BufferCopy copyRegion(0, 0, size);
		commandBuffers[0]->copyBuffer(srcBuffer, dstBuffer, copyRegion);
		
		endSingleTimeCommands(commandBuffers);
	}

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		}

		throw std::runtime_error("failed to find suitable memory type!");
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

			commandBuffers[i]->bindVertexBuffers(0, {vertexBuffer.get()}, {0});

			commandBuffers[i]->bindIndexBuffer(indexBuffer.get(), 0, vk::IndexType::eUint16);

			commandBuffers[i]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, 1, &descriptorSets[i], 0, nullptr);

			commandBuffers[i]->drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

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

	void updateUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data = device->mapMemory(uniformBuffersMemory[currentImage].get(), 0, sizeof(ubo), vk::MemoryMapFlags());
		memcpy(data, &ubo, sizeof(ubo));
		device->unmapMemory(uniformBuffersMemory[currentImage].get());
	}

	void drawFrame() {
		device->waitForFences(1, &inFlightFences[currentFrame].get(), true, UINT64_MAX);

		vk::ResultValue<uint32_t> result = device->acquireNextImageKHR(swapchain.get(), (uint64_t)UINT64_MAX, imageAvailableSemaphores[currentFrame].get(), nullptr);

		if (result.result == vk::Result::eErrorOutOfDateKHR) {
			recreateSwapchain();
			return;
		}
		
		if (imagesInFlight[result.value] != -1) 
			device->waitForFences(1, &inFlightFences[imagesInFlight[result.value]].get(), true, UINT64_MAX);
		imagesInFlight[result.value] = currentFrame;
		
		vk::PipelineStageFlags waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

		device->resetFences(1, &inFlightFences[currentFrame].get());

		updateUniformBuffer(result.value);

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

	void recreateSwapchain() {		
		device->waitIdle();

		createSwapchain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();

		commandBuffers.clear();
		uniformBuffers.clear();
		uniformBuffersMemory.clear();

		createCommandPool();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
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
			int width, height;
			SDL_GetWindowSize(window, &width, &height);

			vk::Extent2D actualExtent = {(uint32_t) width, (uint32_t) height};

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

		vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

		return indices.isComplete() && swapchainAdequate && supportedFeatures.samplerAnisotropy;
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