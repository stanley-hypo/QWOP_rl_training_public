// 自定义的三种游戏模型游戏规则，为了冲榜一致，尽可能不要修改此代码
#include "course_rules.h"

namespace {
constexpr float kFirstHurdleDistance = 13.72f;
constexpr float kHurdleSpacing = 9.14f;
constexpr int kHurdleCount = 10;
constexpr float kClassicHurdleDistance = 50.0f;

std::vector<float> standardHurdleLocations() {
    std::vector<float> locations;
    locations.reserve(kHurdleCount);
    for (int i = 0; i < kHurdleCount; ++i) {
        const float distance = kFirstHurdleDistance + static_cast<float>(i) * kHurdleSpacing;
        locations.push_back(distance * kReferenceUnitsPerRaceMeter);
    }
    return locations;
}
}

CourseSpec makeCourseSpec(GameMode mode) {
    CourseSpec spec;
    switch (mode) {
    case GameMode::Hurdle110m:
        spec.finishType = FinishType::FinishLine;
        spec.observationLayout = ObservationLayout::Hurdle;
        spec.raceDistanceMeters = 110.0f;
        spec.finishLocation = 110.0f * kReferenceUnitsPerRaceMeter;
        spec.hurdleLocations = standardHurdleLocations();
        spec.timeLimit = 120.0f;
        break;
    case GameMode::LongJump30m:
        spec.finishType = FinishType::JumpLanding;
        spec.observationLayout = ObservationLayout::LongJump;
        spec.raceDistanceMeters = kLongJumpSandLength;
        spec.takeoffLineLocation = kLongJumpTakeoffDistance * kReferenceUnitsPerRaceMeter;
        spec.finishLocation = spec.takeoffLineLocation + kLongJumpSandLength * kReferenceUnitsPerRaceMeter;
        spec.sandPitLocation = spec.takeoffLineLocation;
        spec.sandPitEndLocation = spec.finishLocation;
        spec.timeLimit = 45.0f;
        break;
    case GameMode::Classic100m:
    default:
        spec.finishType = FinishType::JumpLanding;
        spec.observationLayout = ObservationLayout::Classic;
        spec.raceDistanceMeters = 100.0f;
        spec.finishLocation = 100.0f * kReferenceUnitsPerRaceMeter;
        spec.hurdleLocations = {kClassicHurdleDistance * kReferenceUnitsPerRaceMeter};
        spec.sandPitLocation = spec.finishLocation;
        spec.sandPitEndLocation = spec.finishLocation;
        spec.timeLimit = 120.0f;
        break;
    }
    return spec;
}

const char* gameModeName(GameMode mode) {
    switch (mode) {
    case GameMode::Hurdle110m:
        return "hurdle_110m";
    case GameMode::LongJump30m:
        return "long_jump_30m";
    case GameMode::Classic100m:
    default:
        return "classic_100m";
    }
}

GameMode inferGameMode(RewardProfile profile) {
    switch (profile) {
    case RewardProfile::Hurdle110m:
        return GameMode::Hurdle110m;
    case RewardProfile::LongJump30m:
        return GameMode::LongJump30m;
    case RewardProfile::Classic100m:
    default:
        return GameMode::Classic100m;
    }
}
