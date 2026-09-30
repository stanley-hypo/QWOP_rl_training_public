#pragma once

#include "course_rules.h"
#include "physics_backend_api.h"

#include <vector>

class RewardState {
public:
    void reset(const CourseSpec& course);
    float calculate(RewardProfile profile, const CourseSpec& course, const RgFrame& frame, bool truncated);

private:
    struct HurdleRewardState {
        bool leadFootApplied = false;
        bool bodyApplied = false;
        bool rearFootApplied = false;
    };

    std::vector<HurdleRewardState> hurdles_;
    float progressScore_ = 0.0f;
    float lastProgress_ = 0.0f;
    float lastTime_ = 0.0f;
    bool longJumpLandingSeen_ = false;
    float longJumpPostLandingX_ = 0.0f;
};

