#include "training_env.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr float kPi = 3.14159265358979323846f;

float degreesToRadians(float degrees) {
    return degrees * kPi / 180.0f;
}
}

TrainingEnv::TrainingEnv()
    : course_(makeCourseSpec(gameMode_)) {
    resetEpisode();
}

EnvStep TrainingEnv::resetEpisode() {
    backend_.reset(backendGameMode(gameMode_));
    frame_ = backend_.readFrame();
    rewardState_.reset(course_);
    return makeStepResult();
}

EnvStep TrainingEnv::stepAction(EnvAction action) {
    backend_.step(actionMask(action));
    frame_ = backend_.readFrame();
    return makeStepResult();
}

void TrainingEnv::setRewardProfile(RewardProfile profile) {
    rewardProfile_ = profile;
}

void TrainingEnv::setGameMode(GameMode mode) {
    if (gameMode_ == mode) {
        return;
    }
    gameMode_ = mode;
    course_ = makeCourseSpec(gameMode_);
    resetEpisode();
}

GameMode TrainingEnv::gameMode() const {
    return gameMode_;
}

size_t TrainingEnv::observationSize() const {
    constexpr size_t bodyValues = RG_PHYSICS_BODY_COUNT * 5;
    return bodyValues + (course_.observationLayout == ObservationLayout::Hurdle ? 6 : 8);
}

uint8_t TrainingEnv::actionMask(EnvAction action) {
    return static_cast<uint8_t>(
        (action.q ? 1 : 0)
        | (action.w ? 2 : 0)
        | (action.o ? 4 : 0)
        | (action.p ? 8 : 0));
}

uint32_t TrainingEnv::backendGameMode(GameMode mode) {
    switch (mode) {
    case GameMode::Hurdle110m:
        return RG_GAME_MODE_HURDLE_110M;
    case GameMode::LongJump30m:
        return RG_GAME_MODE_LONG_JUMP_30M;
    case GameMode::Classic100m:
    default:
        return RG_GAME_MODE_CLASSIC_100M;
    }
}

bool TrainingEnv::isTerminal() const {
    if (frame_.failed) {
        return true;
    }
    if (course_.finishType == FinishType::JumpLanding) {
        return frame_.jump_landed != 0;
    }
    return frame_.race_finished != 0;
}

bool TrainingEnv::isLongJumpMode() const {
    return gameMode_ == GameMode::LongJump30m;
}

EnvStep TrainingEnv::makeStepResult() {
    EnvStep result;
    result.observation = observation();
    result.score = frame_.score;
    result.distance = isLongJumpMode() && frame_.long_jump_landing_started
        ? kLongJumpTakeoffDistance + frame_.long_jump_distance
        : frame_.score;
    result.time = frame_.sim_time;
    result.done = isTerminal();
    result.truncated = frame_.sim_time >= course_.timeLimit;
    result.success = result.done && !result.truncated && !frame_.failed;
    result.reward = rewardState_.calculate(rewardProfile_, course_, frame_, result.truncated);
    return result;
}

std::vector<float> TrainingEnv::observation() const {
    if (frame_.body_count != RG_PHYSICS_BODY_COUNT) {
        throw std::runtime_error("Physics frame body count changed.");
    }

    std::vector<float> obs(observationSize(), 0.0f);
    size_t out = 0;
    for (const RgBodySample& body : frame_.bodies) {
        obs[out++] = (body.x - frame_.view_target_x) / kReferenceUnitsPerPhysicsUnit;
        obs[out++] = body.y / kReferenceUnitsPerPhysicsUnit;
        obs[out++] = degreesToRadians(body.angle_degrees);
        obs[out++] = body.velocity_x;
        obs[out++] = body.velocity_y;
    }

    const float nextHurdleDx = frame_.hurdle_location - frame_.view_target_x;
    obs[out++] = frame_.score;
    obs[out++] = frame_.sim_time;
    obs[out++] = nextHurdleDx / kReferenceUnitsPerPhysicsUnit;
    if (course_.observationLayout == ObservationLayout::LongJump) {
        const float takeoffDx = course_.takeoffLineLocation - frame_.view_target_x;
        const float sandEndDx = course_.sandPitEndLocation - frame_.view_target_x;
        obs[out - 1] = takeoffDx / kReferenceUnitsPerPhysicsUnit;
        obs[out++] = sandEndDx / kReferenceUnitsPerPhysicsUnit;
        obs[out++] = frame_.failed ? 1.0f : 0.0f;
        obs[out++] = frame_.jumped ? 1.0f : 0.0f;
        obs[out++] = frame_.left_foot_grounded ? 1.0f : 0.0f;
        obs[out++] = frame_.right_foot_grounded ? 1.0f : 0.0f;
    } else if (course_.observationLayout == ObservationLayout::Classic) {
        const float sandPitDx = course_.sandPitLocation - frame_.view_target_x;
        obs[out++] = sandPitDx / kReferenceUnitsPerPhysicsUnit;
        obs[out++] = frame_.failed ? 1.0f : 0.0f;
        obs[out++] = frame_.jumped ? 1.0f : 0.0f;
        obs[out++] = frame_.left_foot_grounded ? 1.0f : 0.0f;
        obs[out++] = frame_.right_foot_grounded ? 1.0f : 0.0f;
    } else {
        const float finishDx = course_.finishLocation - frame_.view_target_x;
        obs[out++] = finishDx / kReferenceUnitsPerPhysicsUnit;
        obs[out++] = frame_.left_foot_grounded ? 1.0f : 0.0f;
        obs[out++] = frame_.right_foot_grounded ? 1.0f : 0.0f;
    }
    return obs;
}
