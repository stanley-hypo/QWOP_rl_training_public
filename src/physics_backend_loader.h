#pragma once

#include "physics_backend_api.h"

#include <cstdint>

class PhysicsBackend {
public:
    PhysicsBackend();
    ~PhysicsBackend();

    PhysicsBackend(const PhysicsBackend&) = delete;
    PhysicsBackend& operator=(const PhysicsBackend&) = delete;

    void reset(uint32_t gameMode);
    void step(uint8_t actionMask);
    RgFrame readFrame() const;

private:
    RgHandle handle_ = nullptr;
};
