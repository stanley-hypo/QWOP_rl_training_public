#pragma once

#include "course_rules.h"
#include "physics_backend_loader.h"
#include "reward_profiles.h"

#include <cstddef>
#include <vector>

struct EnvAction {
    bool q = false;
    bool w = false;
    bool o = false;
    bool p = false;
};

struct EnvStep {
    std::vector<float> observation;
    float reward = 0.0f;
    bool done = false;
    bool truncated = false;
    bool success = false;
    float score = 0.0f;
    float distance = 0.0f;
    float time = 0.0f;
};

class TrainingEnv {
public:
    TrainingEnv();

    EnvStep resetEpisode();
    EnvStep stepAction(EnvAction action);
    void setRewardProfile(RewardProfile profile);
    void setGameMode(GameMode mode);
    GameMode gameMode() const;
    size_t observationSize() const;

private:
    static uint8_t actionMask(EnvAction action);
    static uint32_t backendGameMode(GameMode mode);
    bool isTerminal() const;
    bool isLongJumpMode() const;
    EnvStep makeStepResult();
    std::vector<float> observation() const;

    PhysicsBackend backend_;
    RgFrame frame_{};
    GameMode gameMode_ = GameMode::Classic100m;
    RewardProfile rewardProfile_ = RewardProfile::Classic100m;
    CourseSpec course_;
    RewardState rewardState_;
};
