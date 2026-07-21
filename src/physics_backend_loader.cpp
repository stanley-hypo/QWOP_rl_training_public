#include "physics_backend_loader.h"

#ifdef _WIN32
#include <windows.h>
#else
#error The prebuilt physics backend currently supports Windows only.
#endif

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
int gModuleAnchor = 0;

using CreateFn = RgResult (*)(RgHandle*);
using DestroyFn = void (*)(RgHandle);
using ResetFn = RgResult (*)(RgHandle, const RgCourse*);
using StepFn = RgResult (*)(RgHandle, uint8_t);
using ReadFrameFn = RgResult (*)(RgHandle, RgFrame*);
using LastErrorFn = const char* (*)(RgHandle);

std::filesystem::path moduleDirectory() {
    HMODULE module = nullptr;
    const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&gModuleAnchor), &module)) {
        throw std::runtime_error("Cannot locate the _running_env module directory.");
    }

    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) {
        throw std::runtime_error("Cannot resolve the _running_env module path.");
    }
    path.resize(length);
    return std::filesystem::path(path).parent_path();
}

std::filesystem::path backendPath() {
    if (const char* configured = std::getenv("RUNNING_PHYSICS_DLL")) {
        if (*configured != '\0') {
            return std::filesystem::absolute(configured);
        }
    }
    return moduleDirectory() / "running_physics.dll";
}

template <typename Fn>
Fn loadFunction(HMODULE module, const char* name) {
    FARPROC address = GetProcAddress(module, name);
    if (!address) {
        throw std::runtime_error(std::string("running_physics.dll is missing export: ") + name);
    }
    return reinterpret_cast<Fn>(address);
}

struct BackendModule {
    HMODULE module = nullptr;
    CreateFn create = nullptr;
    DestroyFn destroy = nullptr;
    ResetFn reset = nullptr;
    StepFn step = nullptr;
    ReadFrameFn readFrame = nullptr;
    LastErrorFn lastError = nullptr;

    BackendModule() {
        const std::filesystem::path path = backendPath();
        module = LoadLibraryExW(
            path.c_str(),
            nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module) {
            throw std::runtime_error("Cannot load physics backend: " + path.string());
        }

        create = loadFunction<CreateFn>(module, "rg_create");
        destroy = loadFunction<DestroyFn>(module, "rg_destroy");
        reset = loadFunction<ResetFn>(module, "rg_reset");
        step = loadFunction<StepFn>(module, "rg_step");
        readFrame = loadFunction<ReadFrameFn>(module, "rg_read_frame");
        lastError = loadFunction<LastErrorFn>(module, "rg_last_error");

    }
};

BackendModule& backendModule() {
    static BackendModule module;
    return module;
}

void checkResult(RgResult result, RgHandle handle, const char* operation) {
    if (result == RG_RESULT_OK) {
        return;
    }
    const char* detail = backendModule().lastError(handle);
    throw std::runtime_error(
        std::string("Physics backend ") + operation + " failed"
        + (detail && *detail ? std::string(": ") + detail : std::string()));
}
}

PhysicsBackend::PhysicsBackend() {
    checkResult(backendModule().create(&handle_), nullptr, "create");
    if (!handle_) {
        throw std::runtime_error("Physics backend returned a null handle.");
    }
}

PhysicsBackend::~PhysicsBackend() {
    if (handle_) {
        backendModule().destroy(handle_);
    }
}

void PhysicsBackend::reset(uint32_t gameMode) {
    const RgCourse course{sizeof(RgCourse), gameMode};
    checkResult(backendModule().reset(handle_, &course), handle_, "reset");
}

void PhysicsBackend::step(uint8_t actionMask) {
    checkResult(backendModule().step(handle_, actionMask), handle_, "step");
}

RgFrame PhysicsBackend::readFrame() const {
    RgFrame frame{};
    frame.struct_size = sizeof(RgFrame);
    checkResult(backendModule().readFrame(handle_, &frame), handle_, "read_frame");
    if (frame.body_count != RG_PHYSICS_BODY_COUNT) {
        throw std::runtime_error("Physics backend returned an unsupported body count.");
    }
    return frame;
}
