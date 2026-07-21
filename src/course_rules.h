#pragma once

#include <cstddef>
#include <vector>

enum class GameMode {
    Classic100m,
    Hurdle110m,
    LongJump30m,
};

enum class RewardProfile {
    Classic100m,
    Hurdle110m,
    LongJump30m,
};

enum class FinishType {
    JumpLanding,
    FinishLine,
};

enum class ObservationLayout {
    Classic,
    Hurdle,
    LongJump,
};

struct CourseSpec {
    FinishType finishType = FinishType::JumpLanding;
    ObservationLayout observationLayout = ObservationLayout::Classic;
    float raceDistanceMeters = 100.0f;
    float finishLocation = 0.0f;
    std::vector<float> hurdleLocations;
    float sandPitLocation = 0.0f;
    float sandPitEndLocation = 0.0f;
    float takeoffLineLocation = 0.0f;
    float timeLimit = 120.0f;
};

inline constexpr float kReferenceUnitsPerPhysicsUnit = 40.0f;
inline constexpr float kReferenceUnitsPerRaceMeter = 400.0f;
inline constexpr float kLongJumpTakeoffDistance = 30.0f;
inline constexpr float kLongJumpSandLength = 9.0f;
inline constexpr size_t kLeftFootBodySlot = 3;
inline constexpr size_t kTorsoBodySlot = 5;
inline constexpr size_t kHeadBodySlot = 10;
inline constexpr size_t kRightFootBodySlot = 11;

CourseSpec makeCourseSpec(GameMode mode);
const char* gameModeName(GameMode mode);
GameMode inferGameMode(RewardProfile profile);

