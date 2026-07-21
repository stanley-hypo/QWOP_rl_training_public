#include "training_env.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

RewardProfile rewardProfileFromName(const std::string& name) {
    if (name == "classic_100m") {
        return RewardProfile::Classic100m;
    }
    if (name == "hurdle_110m") {
        return RewardProfile::Hurdle110m;
    }
    if (name == "long_jump_30m") {
        return RewardProfile::LongJump30m;
    }
    throw std::invalid_argument("unknown reward profile: " + name);
}

GameMode gameModeFromName(const std::string& name, RewardProfile profile) {
    if (name.empty() || name == "auto") {
        return inferGameMode(profile);
    }
    GameMode mode;
    if (name == "classic_100m") {
        mode = GameMode::Classic100m;
    } else if (name == "hurdle_110m") {
        mode = GameMode::Hurdle110m;
    } else if (name == "long_jump_30m") {
        mode = GameMode::LongJump30m;
    } else {
        throw std::invalid_argument("unknown game mode: " + name);
    }
    if (mode != inferGameMode(profile)) {
        throw std::invalid_argument("game mode does not match reward profile: " + name);
    }
    return mode;
}

EnvAction actionFromMask(int actionMask) {
    return EnvAction{
        (actionMask & 1) != 0,
        (actionMask & 2) != 0,
        (actionMask & 4) != 0,
        (actionMask & 8) != 0,
    };
}

std::vector<float> observationVector(const EnvStep& step) {
    return step.observation;
}

py::dict stepToDict(const EnvStep& step) {
    py::dict out;
    out["obs"] = observationVector(step);
    out["reward"] = step.reward;
    out["done"] = step.done;
    out["truncated"] = step.truncated;
    out["success"] = step.success;
    out["score"] = step.score;
    out["distance"] = step.distance;
    out["time"] = step.time;
    return out;
}

class BatchedRunningEnv {
public:
    explicit BatchedRunningEnv(
        size_t numEnvs,
        size_t numThreads = 1,
        const std::string& rewardProfile = "classic_100m",
        const std::string& gameMode = "auto")
        : envs_(numEnvs),
          threadCount_(std::max<size_t>(1, std::min(numThreads, numEnvs))),
          rewards_(numEnvs),
          terminated_(numEnvs),
          truncated_(numEnvs),
          scores_(numEnvs),
          times_(numEnvs),
          finalScores_(numEnvs),
          finalDistances_(numEnvs),
          finalTimes_(numEnvs),
          finalSuccess_(numEnvs),
          resetScores_(numEnvs),
          resetTimes_(numEnvs),
          hasFinal_(numEnvs) {
        if (numEnvs == 0) {
            throw std::invalid_argument("num_envs must be positive");
        }
        const RewardProfile profile = rewardProfileFromName(rewardProfile);
        const GameMode mode = gameModeFromName(gameMode, profile);
        gameMode_ = mode;
        for (TrainingEnv& env : envs_) {
            env.setGameMode(mode);
            env.setRewardProfile(profile);
        }
        observationSize_ = envs_.front().observationSize();
        observations_.assign(numEnvs * observationSize_, 0.0f);
        startWorkers();
    }

    ~BatchedRunningEnv() {
        stopWorkers();
    }

    py::dict reset() {
        {
            py::gil_scoped_release release;
            runParallel([this](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    EnvStep step = envs_[i].resetEpisode();
                    writeObservation(i, step.observation, observations_, observationSize_);
                    rewards_[i] = 0.0f;
                    terminated_[i] = 0;
                    truncated_[i] = 0;
                    scores_[i] = step.score;
                    times_[i] = step.time;
                    finalScores_[i] = 0.0f;
                    finalDistances_[i] = 0.0f;
                    finalTimes_[i] = 0.0f;
                    finalSuccess_[i] = 0;
                    resetScores_[i] = step.score;
                    resetTimes_[i] = step.time;
                    hasFinal_[i] = 0;
                }
            });
        }
        return resultDict();
    }

    py::dict step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions) {
        py::buffer_info actionInfo = actions.request();
        if (actionInfo.ndim != 1 || static_cast<size_t>(actionInfo.shape[0]) != envs_.size()) {
            throw py::value_error("actions shape must be (num_envs,)");
        }

        const auto* actionData = static_cast<const int64_t*>(actionInfo.ptr);
        {
            py::gil_scoped_release release;
            runParallel([this, actionData](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    EnvStep step = envs_[i].stepAction(actionFromMask(static_cast<int>(actionData[i])));
                    rewards_[i] = step.reward;
                    terminated_[i] = step.done ? 1 : 0;
                    truncated_[i] = step.truncated ? 1 : 0;
                    scores_[i] = step.score;
                    times_[i] = step.time;
                    hasFinal_[i] = (step.done || step.truncated) ? 1 : 0;

                    if (hasFinal_[i]) {
                        finalScores_[i] = step.score;
                        finalDistances_[i] = step.distance;
                        finalTimes_[i] = step.time;
                        finalSuccess_[i] = step.success ? 1 : 0;

                        EnvStep resetStep = envs_[i].resetEpisode();
                        writeObservation(i, resetStep.observation, observations_, observationSize_);
                        resetScores_[i] = resetStep.score;
                        resetTimes_[i] = resetStep.time;
                    } else {
                        writeObservation(i, step.observation, observations_, observationSize_);
                        finalScores_[i] = 0.0f;
                        finalDistances_[i] = 0.0f;
                        finalTimes_[i] = 0.0f;
                        finalSuccess_[i] = 0;
                        resetScores_[i] = 0.0f;
                        resetTimes_[i] = 0.0f;
                    }
                }
            });
        }

        return resultDict();
    }

    size_t numEnvs() const {
        return envs_.size();
    }

    size_t numThreads() const {
        return threadCount_;
    }

    size_t observationSize() const {
        return observationSize_;
    }

    std::string gameModeNameValue() const {
        return std::string(gameModeName(gameMode_));
    }

private:
    struct WorkRange {
        size_t begin = 0;
        size_t end = 0;
    };

    void startWorkers() {
        if (threadCount_ <= 1) {
            return;
        }

        ranges_.resize(threadCount_);
        workers_.reserve(threadCount_);
        for (size_t workerIndex = 0; workerIndex < threadCount_; ++workerIndex) {
            workers_.emplace_back([this, workerIndex] {
                workerLoop(workerIndex);
            });
        }
    }

    void stopWorkers() {
        {
            std::lock_guard<std::mutex> lock(workMutex_);
            stopping_ = true;
            ++workGeneration_;
        }
        workCv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    template <typename Fn>
    void runParallel(Fn&& fn) {
        if (threadCount_ <= 1) {
            fn(0, envs_.size());
            return;
        }

        currentWork_ = [&fn](size_t begin, size_t end) {
            fn(begin, end);
        };

        const size_t count = envs_.size();
        const size_t chunk = (count + threadCount_ - 1) / threadCount_;
        {
            std::lock_guard<std::mutex> lock(workMutex_);
            remainingWorkers_ = threadCount_;
            for (size_t i = 0; i < threadCount_; ++i) {
                const size_t begin = std::min(i * chunk, count);
                const size_t end = std::min(begin + chunk, count);
                ranges_[i] = WorkRange{begin, end};
            }
            ++workGeneration_;
        }

        workCv_.notify_all();
        std::unique_lock<std::mutex> lock(workMutex_);
        doneCv_.wait(lock, [this] {
            return remainingWorkers_ == 0;
        });
        currentWork_ = nullptr;
    }

    void workerLoop(size_t workerIndex) {
        size_t seenGeneration = 0;
        while (true) {
            WorkRange range;
            std::function<void(size_t, size_t)> work;
            {
                std::unique_lock<std::mutex> lock(workMutex_);
                workCv_.wait(lock, [this, &seenGeneration] {
                    return stopping_ || workGeneration_ != seenGeneration;
                });
                if (stopping_) {
                    return;
                }
                seenGeneration = workGeneration_;
                range = ranges_[workerIndex];
                work = currentWork_;
            }

            if (work && range.begin < range.end) {
                work(range.begin, range.end);
            }

            {
                std::lock_guard<std::mutex> lock(workMutex_);
                --remainingWorkers_;
                if (remainingWorkers_ == 0) {
                    doneCv_.notify_one();
                }
            }
        }
    }

    static void writeObservation(
        size_t envIndex,
        const std::vector<float>& observation,
        std::vector<float>& out,
        size_t observationSize) {
        if (observation.size() != observationSize) {
            throw std::runtime_error("observation size changed inside a batch");
        }
        const size_t offset = envIndex * observationSize;
        std::copy(observation.begin(), observation.end(), out.begin() + static_cast<std::ptrdiff_t>(offset));
    }

    template <typename T>
    py::array view1d(std::vector<T>& data) {
        return py::array_t<T>(
            {static_cast<py::ssize_t>(envs_.size())},
            {static_cast<py::ssize_t>(sizeof(T))},
            data.data());
    }

    py::array boolView1d(std::vector<uint8_t>& data) {
        return py::array(
            py::dtype("bool"),
            {static_cast<py::ssize_t>(envs_.size())},
            {static_cast<py::ssize_t>(sizeof(uint8_t))},
            data.data());
    }

    py::array observationView(std::vector<float>& data) {
        return py::array_t<float>(
            {
                static_cast<py::ssize_t>(envs_.size()),
                static_cast<py::ssize_t>(observationSize_),
            },
            {
                static_cast<py::ssize_t>(observationSize_ * sizeof(float)),
                static_cast<py::ssize_t>(sizeof(float)),
            },
            data.data());
    }

    py::dict resultDict() {
        py::dict out;
        out["obs"] = observationView(observations_);
        out["reward"] = view1d(rewards_);
        out["done"] = boolView1d(terminated_);
        out["truncated"] = boolView1d(truncated_);
        out["score"] = view1d(scores_);
        out["time"] = view1d(times_);
        out["final_score"] = view1d(finalScores_);
        out["final_distance"] = view1d(finalDistances_);
        out["final_time"] = view1d(finalTimes_);
        out["final_success"] = boolView1d(finalSuccess_);
        out["reset_score"] = view1d(resetScores_);
        out["reset_time"] = view1d(resetTimes_);
        out["has_final"] = boolView1d(hasFinal_);
        return out;
    }

    std::vector<TrainingEnv> envs_;
    size_t threadCount_ = 1;
    size_t observationSize_ = 0;
    GameMode gameMode_ = GameMode::Classic100m;
    std::vector<float> observations_;
    std::vector<float> rewards_;
    std::vector<uint8_t> terminated_;
    std::vector<uint8_t> truncated_;
    std::vector<float> scores_;
    std::vector<float> times_;
    std::vector<float> finalScores_;
    std::vector<float> finalDistances_;
    std::vector<float> finalTimes_;
    std::vector<uint8_t> finalSuccess_;
    std::vector<float> resetScores_;
    std::vector<float> resetTimes_;
    std::vector<uint8_t> hasFinal_;
    std::vector<std::thread> workers_;
    std::vector<WorkRange> ranges_;
    std::mutex workMutex_;
    std::condition_variable workCv_;
    std::condition_variable doneCv_;
    std::function<void(size_t, size_t)> currentWork_;
    size_t remainingWorkers_ = 0;
    size_t workGeneration_ = 0;
    bool stopping_ = false;
};

} // namespace

PYBIND11_MODULE(_running_env, m) {
    m.doc() = "Running training environment with an external physics backend.";
    py::class_<TrainingEnv>(m, "Env")
        .def(py::init<>())
        .def("configure", [](TrainingEnv& env, const std::string& rewardProfile, const std::string& gameMode) {
            const RewardProfile profile = rewardProfileFromName(rewardProfile);
            env.setGameMode(gameModeFromName(gameMode, profile));
            env.setRewardProfile(profile);
        }, py::arg("reward_profile"), py::arg("game_mode"))
        .def("set_reward_profile", [](TrainingEnv& env, const std::string& rewardProfile) {
            const RewardProfile profile = rewardProfileFromName(rewardProfile);
            env.setGameMode(inferGameMode(profile));
            env.setRewardProfile(profile);
        })
        .def("set_game_mode", [](TrainingEnv& env, const std::string& gameMode, const std::string& rewardProfile) {
            const RewardProfile profile = rewardProfileFromName(rewardProfile);
            env.setGameMode(gameModeFromName(gameMode, profile));
            env.setRewardProfile(profile);
        })
        .def_property_readonly("observation_size", &TrainingEnv::observationSize)
        .def_property_readonly("game_mode", [](const TrainingEnv& env) {
            return std::string(gameModeName(env.gameMode()));
        })
        .def("reset", [](TrainingEnv& env) {
            return stepToDict(env.resetEpisode());
        })
        .def("step", [](TrainingEnv& env, int actionMask) {
            return stepToDict(env.stepAction(actionFromMask(actionMask)));
        });

    py::class_<BatchedRunningEnv>(m, "BatchedEnv")
        .def(
            py::init<size_t, size_t, const std::string&, const std::string&>(),
            py::arg("num_envs"),
            py::arg("num_threads") = 1,
            py::arg("reward_profile") = "classic_100m",
            py::arg("game_mode") = "auto")
        .def("reset", &BatchedRunningEnv::reset)
        .def("step", &BatchedRunningEnv::step)
        .def_property_readonly("num_envs", &BatchedRunningEnv::numEnvs)
        .def_property_readonly("num_threads", &BatchedRunningEnv::numThreads)
        .def_property_readonly("observation_size", &BatchedRunningEnv::observationSize)
        .def_property_readonly("game_mode", &BatchedRunningEnv::gameModeNameValue);

    m.def(
        "observation_size",
        [](const std::string& gameMode, const std::string& rewardProfile) {
            const RewardProfile profile = rewardProfileFromName(rewardProfile);
            TrainingEnv env;
            env.setGameMode(gameModeFromName(gameMode, profile));
            return env.observationSize();
        },
        py::arg("game_mode") = "auto",
        py::arg("reward_profile") = "classic_100m");
    m.def(
        "infer_game_mode",
        [](const std::string& rewardProfile) {
            return std::string(gameModeName(inferGameMode(rewardProfileFromName(rewardProfile))));
        },
        py::arg("reward_profile") = "classic_100m");
}
