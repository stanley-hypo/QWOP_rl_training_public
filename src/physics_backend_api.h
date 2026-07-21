#pragma once

#include <stdint.h>

#ifdef _WIN32
#  ifdef RUNNING_PHYSICS_EXPORTS
#    define RG_PHYSICS_API __declspec(dllexport)
#  else
#    define RG_PHYSICS_API __declspec(dllimport)
#  endif
#else
#  define RG_PHYSICS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    RG_PHYSICS_BODY_COUNT = 12,
};

typedef void* RgHandle;

typedef enum RgResult {
    RG_RESULT_OK = 0,
    RG_RESULT_INVALID_ARGUMENT = 1,
    RG_RESULT_INVALID_STATE = 2,
    RG_RESULT_INTERNAL_ERROR = 3,
} RgResult;

typedef enum RgGameMode {
    RG_GAME_MODE_CLASSIC_100M = 0,
    RG_GAME_MODE_HURDLE_110M = 1,
    RG_GAME_MODE_LONG_JUMP_30M = 2,
} RgGameMode;

typedef struct RgCourse {
    uint32_t struct_size;
    uint32_t game_mode;
} RgCourse;

typedef struct RgBodySample {
    float x;
    float y;
    float angle_degrees;
    float velocity_x;
    float velocity_y;
} RgBodySample;

typedef struct RgFrame {
    uint32_t struct_size;
    uint32_t body_count;
    RgBodySample bodies[RG_PHYSICS_BODY_COUNT];
    float score;
    float sim_time;
    float view_target_x;
    float hurdle_location;
    float long_jump_distance;
    float long_jump_landing_body_x;
    uint32_t active_hurdle_count;
    uint8_t failed;
    uint8_t jumped;
    uint8_t jump_landed;
    uint8_t race_finished;
    uint8_t left_foot_grounded;
    uint8_t right_foot_grounded;
    uint8_t long_jump_landing_started;
    uint8_t long_jump_exited_sand;
} RgFrame;

RG_PHYSICS_API RgResult rg_create(RgHandle* out_handle);
RG_PHYSICS_API void rg_destroy(RgHandle handle);
RG_PHYSICS_API RgResult rg_reset(RgHandle handle, const RgCourse* course);
RG_PHYSICS_API RgResult rg_step(RgHandle handle, uint8_t action_mask);
RG_PHYSICS_API RgResult rg_read_frame(RgHandle handle, RgFrame* out_frame);
RG_PHYSICS_API const char* rg_last_error(RgHandle handle);

#ifdef __cplusplus
}
#endif
