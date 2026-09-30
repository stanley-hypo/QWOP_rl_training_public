#include "reward_profiles.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr float kTrackY = 429.8f;
constexpr float kHurdleRewardHeight = 381.5f;
constexpr float kStandingBodyY = -77.05f;
constexpr float kLeadFootHeightRewardCap = 8.0f;
constexpr float kLeadFootHeightCapScale = 1.1f;
constexpr float kRearFootHeightRewardCap = 8.0f;
constexpr float kBodyHeightRewardCap = 5.0f;
constexpr float kBodyHeightTargetHurdleScale = 0.45f;
constexpr float kLongJumpLandingDistanceScale = 5.0f;
constexpr float kLongJumpPostLandingProgressScale = 0.5f;

float scoreFromReferenceX(float x) {
    return std::round(x / kReferenceUnitsPerPhysicsUnit) / 10.0f;
}

float heightRewardRatio(float y) {
    const float height = std::max(0.0f, kTrackY - y);
    return std::clamp(height / (kHurdleRewardHeight * kLeadFootHeightCapScale), 0.0f, 1.0f);
}

float bodyHeightRewardRatio(float bodyY) {
    const float targetLift = kHurdleRewardHeight * kBodyHeightTargetHurdleScale;
    const float bodyLift = std::max(0.0f, kStandingBodyY - bodyY);
    return std::clamp(bodyLift / targetLift, 0.0f, 1.0f);
}
}

void RewardState::reset(const CourseSpec& course) {
    hurdles_.assign(course.hurdleLocations.size(), HurdleRewardState{});
    progressScore_ = 0.0f;
    lastProgress_ = 0.0f;
    lastTime_ = 0.0f;
    longJumpLandingSeen_ = false;
    longJumpPostLandingX_ = 0.0f;
}

float RewardState::calculate(
    RewardProfile profile,
    const CourseSpec& course,
    const RgFrame& frame,
    bool truncated) {
    switch (profile) {
    case RewardProfile::Classic100m: {
        const float cappedProgress = std::clamp(frame.score, 0.0f, 100.0f);
        float reward = std::max(0.0f, cappedProgress - progressScore_);
        progressScore_ = std::max(progressScore_, cappedProgress);
        // v2 velocity shaping: reward real forward speed to escape the slow-shuffle optimum
        const float dt = std::max(0.001f, frame.sim_time - lastTime_);
        const float velocity = std::max(0.0f, cappedProgress - lastProgress_) / dt;
        reward += 0.05f * velocity;
        if (velocity > 2.5f) reward += 0.02f;
        lastProgress_ = cappedProgress;
        lastTime_ = frame.sim_time;
        if (frame.failed || truncated) {
            reward -= 10.0f;
        } else if (frame.jump_landed) {
            reward += 120.0f - frame.sim_time;
        }
        return reward;
    }
    case RewardProfile::Hurdle110m: {
        const float cappedProgress = std::clamp(frame.score, 0.0f, 110.0f);
        float reward = std::max(0.0f, cappedProgress - progressScore_);
        progressScore_ = std::max(progressScore_, cappedProgress);

        if (!frame.failed) {
            const size_t activeCount = std::min<size_t>(frame.active_hurdle_count, hurdles_.size());
            for (size_t i = 0; i < activeCount; ++i) {
                HurdleRewardState& hurdle = hurdles_[i];
                const float location = course.hurdleLocations[i];
                const RgBodySample& leadFoot = frame.bodies[kLeftFootBodySlot];
                const RgBodySample& rearFoot = frame.bodies[kRightFootBodySlot];
                const RgBodySample& body = frame.bodies[kTorsoBodySlot];
                if (!hurdle.leadFootApplied && leadFoot.x >= location) {
                    hurdle.leadFootApplied = true;
                    reward += heightRewardRatio(leadFoot.y) * kLeadFootHeightRewardCap;
                }
                if (!hurdle.bodyApplied && body.x >= location) {
                    hurdle.bodyApplied = true;
                    reward += bodyHeightRewardRatio(body.y) * kBodyHeightRewardCap;
                }
                if (!hurdle.rearFootApplied && rearFoot.x >= location) {
                    hurdle.rearFootApplied = true;
                    reward += heightRewardRatio(rearFoot.y) * kRearFootHeightRewardCap;
                }
            }
        }
        if (frame.failed) {
            reward -= 10.0f;
        } else if (frame.race_finished) {
            reward += 120.0f - frame.sim_time;
        }
        return reward;
    }
    case RewardProfile::LongJump30m: {
        const float runupProgress = std::clamp(
            scoreFromReferenceX(frame.view_target_x),
            0.0f,
            kLongJumpTakeoffDistance);
        float reward = std::max(0.0f, runupProgress - progressScore_);
        progressScore_ = std::max(progressScore_, runupProgress);

        if (frame.long_jump_landing_started) {
            if (!longJumpLandingSeen_) {
                longJumpLandingSeen_ = true;
                longJumpPostLandingX_ = frame.long_jump_landing_body_x;
            }
            const float currentX = std::min(frame.bodies[kTorsoBodySlot].x, course.sandPitEndLocation);
            const float walkingDistance = std::max(
                0.0f,
                (currentX - longJumpPostLandingX_) / kReferenceUnitsPerRaceMeter);
            reward += walkingDistance * kLongJumpPostLandingProgressScale;
            longJumpPostLandingX_ = std::max(longJumpPostLandingX_, currentX);
        }
        if (frame.long_jump_exited_sand) {
            reward += frame.long_jump_distance * kLongJumpLandingDistanceScale;
        }
        if (frame.failed) {
            reward -= 10.0f;
        }
        return reward;
    }
    }
    return 0.0f;
}
