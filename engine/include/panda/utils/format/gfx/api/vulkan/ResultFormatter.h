#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <fmt/base.h>
#include <fmt/format.h>
#include <vulkan/vulkan_core.h>

#include <string_view>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

template <>
struct fmt::formatter<vk::Result> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(vk::Result result, FormatContext& ctx) const -> decltype(ctx.out())
    {
        return formatter<std::string_view>::format(getResultName(result), ctx);
    }

    [[nodiscard]] static constexpr auto getResultName(vk::Result result) noexcept -> std::string_view
    {
        using namespace std::string_view_literals;
        switch (result)
        {
        case vk::Result::eSuccess:
            return "Success"sv;
        case vk::Result::eNotReady:
            return "NotReady"sv;
        case vk::Result::eTimeout:
            return "Timeout"sv;
        case vk::Result::eEventSet:
            return "EventSet"sv;
        case vk::Result::eEventReset:
            return "EventReset"sv;
        case vk::Result::eIncomplete:
            return "Incomplete"sv;
        case vk::Result::eErrorOutOfHostMemory:
            return "ErrorOutOfHostMemory"sv;
        case vk::Result::eErrorOutOfDeviceMemory:
            return "ErrorOutOfDeviceMemory"sv;
        case vk::Result::eErrorInitializationFailed:
            return "ErrorInitializationFailed"sv;
        case vk::Result::eErrorDeviceLost:
            return "ErrorDeviceLost"sv;
        case vk::Result::eErrorMemoryMapFailed:
            return "ErrorMemoryMapFailed"sv;
        case vk::Result::eErrorLayerNotPresent:
            return "ErrorLayerNotPresent"sv;
        case vk::Result::eErrorExtensionNotPresent:
            return "ErrorExtensionNotPresent"sv;
        case vk::Result::eErrorFeatureNotPresent:
            return "ErrorFeatureNotPresent"sv;
        case vk::Result::eErrorIncompatibleDriver:
            return "ErrorIncompatibleDriver"sv;
        case vk::Result::eErrorTooManyObjects:
            return "ErrorTooManyObjects"sv;
        case vk::Result::eErrorFormatNotSupported:
            return "ErrorFormatNotSupported"sv;
        case vk::Result::eErrorFragmentedPool:
            return "ErrorFragmentedPool"sv;
        case vk::Result::eErrorUnknown:
            return "ErrorUnknown"sv;
        case vk::Result::eErrorOutOfPoolMemory:
            return "ErrorOutOfPoolMemory"sv;
        case vk::Result::eErrorInvalidExternalHandle:
            return "ErrorInvalidExternalHandle"sv;
        case vk::Result::eErrorFragmentation:
            return "ErrorFragmentation"sv;
        case vk::Result::eErrorInvalidOpaqueCaptureAddress:
            return "ErrorInvalidOpaqueCaptureAddress"sv;
        case vk::Result::ePipelineCompileRequired:
            return "PipelineCompatibleRequired"sv;
        case vk::Result::eErrorSurfaceLostKHR:
            return "ErrorSurfaceLostKHR"sv;
        case vk::Result::eErrorNativeWindowInUseKHR:
            return "ErrorNativeWindowInUseKHR"sv;
        case vk::Result::eSuboptimalKHR:
            return "SuboptimalKHR"sv;
        case vk::Result::eErrorOutOfDateKHR:
            return "ErrorOutOfDateKHR"sv;
        case vk::Result::eErrorIncompatibleDisplayKHR:
            return "ErrorIncompatibleDisplayKHR"sv;
        case vk::Result::eErrorValidationFailedEXT:
            return "ErrorValidationFailedEXT"sv;
        case vk::Result::eErrorInvalidShaderNV:
            return "ErrorInvalidShaderNV"sv;
        case vk::Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT:
            return "ErrorInvalidDrmFormatModifierPlaneLayoutEXT"sv;
        case vk::Result::eErrorNotPermittedKHR:
            return "ErrorNotPermittedKHR"sv;
        case vk::Result::eThreadIdleKHR:
            return "ThreadIdleKHR"sv;
        case vk::Result::eThreadDoneKHR:
            return "ThreadDoneKHR"sv;
        case vk::Result::eOperationDeferredKHR:
            return "OperationDeferredKHR"sv;
        case vk::Result::eOperationNotDeferredKHR:
            return "OperationNotDeferredKHR"sv;
        case vk::Result::eErrorCompressionExhaustedEXT:
            return "ErrorCompressionExhaustedEXT"sv;
        case vk::Result::eErrorImageUsageNotSupportedKHR:
            return "ErrorImageUsageNotSupportedKHR"sv;
        case vk::Result::eErrorVideoPictureLayoutNotSupportedKHR:
            return "ErrorVideoPictureLayoutNotSupportedKHR"sv;
        case vk::Result::eErrorVideoProfileOperationNotSupportedKHR:
            return "ErrorVideoProfileOperationNotSupportedKHR"sv;
        case vk::Result::eErrorVideoProfileFormatNotSupportedKHR:
            return "ErrorVideoProfileFormatNotSupportedKHR"sv;
        case vk::Result::eErrorVideoStdVersionNotSupportedKHR:
            return "ErrorVideoVersionNotSupportedKHR"sv;
        case vk::Result::eErrorVideoProfileCodecNotSupportedKHR:
            return "ErrorVideoProfileCodecNotSupportedKHR"sv;
        default:
            return "UnknownResult"sv;
        }
    }

    [[nodiscard]] static constexpr auto getResultName(VkResult result) noexcept -> std::string_view
    {
        return getResultName(vk::Result {result});
    }
};
