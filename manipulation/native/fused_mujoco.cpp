#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

struct Job {
  const double* initial_state = nullptr;
  const double* initial_ctrl = nullptr;
  const double* controls = nullptr;
  const double* nominal = nullptr;
  const double* ctrl_inv_scale_sq = nullptr;
  double* positions = nullptr;
  double* costs = nullptr;
  double* progress = nullptr;
  bool* failed = nullptr;

  int n = 0;
  int h = 0;
  int nu = 0;
  int substeps = 1;
  int root_qadr = -1;
  int root_body_id = -1;
  int task_qadr = 0;
  int track_kind = 0;  // 0=stadium/closed, 1=straight/open

  double origin_x = 0.0;
  double origin_y = 0.0;
  double rot00 = 1.0;
  double rot01 = 0.0;
  double rot10 = 0.0;
  double rot11 = 1.0;
  double canonical_start_x = 0.0;
  double canonical_start_y = 0.0;
  double radius = 1.0;
  double left_arc_x = 0.0;
  double right_arc_x = 1.0;
  double center_y = 0.0;
  double straight_length = 1.0;
  double track_length = 1.0;
  double allowed_sq = 1.0;
  double min_height = 0.0;
  double min_up = -1.0;
  double upright_weight = 0.0;
  double control_deviation_weight = 1e-4;
  double command_derivative_weight = 0.0;
  double box_progress_weight = 1.0;
  double robot_progress_weight = 0.35;
  double robot_box_approach_weight = 1.00;
  double box_max_lift = 0.035;
  double box_min_up = 0.85;
  double box_lift_weight = 0.0;
  double current_s = 0.0;
  double current_root_s = 0.0;
  double initial_task_root_distance = 0.0;
  double initial_task_height = 0.0;
  double start_time = 0.0;
  double push_standoff = 0.0;
  double goal_s = 0.0;
  double target_error_weight = 0.0;
  double terminal_target_weight = 0.0;
  double overshoot_weight = 20.0;
  double terminal_overshoot_weight = 80.0;
  double target_velocity_weight = 0.0;
  double target_tolerance = 0.005;
  double non_tip_contact_weight = 5.0;
  double tip_contact_reward = 0.0;
  double tip_push_reward = 0.0;
  double table_contact_weight = 50.0;
  double box_speed_weight = 0.0;
  double preferred_box_speed = 0.05;
  double terminal_velocity_weight = 0.0;
  int task_dadr = -1;
  int task_mode = 1;  // 1=push-face, 2=button, 3=reach point, 4=box point push
  double reach_terminal_weight = 0;
  double reach_velocity_weight = 0;
  double reach_region_weight = 0;
  double reach_settle_weight = 0;
  double reach_velocity_ref = 0.1;
  double reach_velocity_sigma = 0.05;
  double reach_safe_radius = 0.015;
  int reach_settle_steps = 30;
  double reach_acceleration_weight = 0;
};

struct SpgJob {
  const double* nominal = nullptr;
  const std::int64_t* ids = nullptr;
  double* jacobian = nullptr;
  double* time_sensitivity = nullptr;

  int nwork = 0;
  int m = 0;
  int h = 0;
  int nu = 0;
  int lookahead = 1;
  int output_steps = 0;
  int substeps = 1;
  double epsilon_fraction = 1e-3;
};

enum class WorkKind {
  kNone,
  kEvaluate,
  kSpg,
};

inline void ProjectStadium(const Job& j, double xw, double yw,
                           double* best_s, double* best_d2) {
  const double dxw = xw - j.origin_x;
  const double dyw = yw - j.origin_y;
  const double px = dxw * j.rot00 + dyw * j.rot10 + j.canonical_start_x;
  const double py = dxw * j.rot01 + dyw * j.rot11 + j.canonical_start_y;

  constexpr double kPi = 3.141592653589793238462643383279502884;
  const double bottom_y = j.center_y - j.radius;
  const double top_y = j.center_y + j.radius;
  const double top_offset = j.straight_length + kPi * j.radius;
  const double left_offset = 2.0 * j.straight_length + kPi * j.radius;

  // Bottom straight.
  const double bx = std::min(std::max(px, j.left_arc_x), j.right_arc_x);
  double ex = px - bx;
  double ey = py - bottom_y;
  double d2 = ex * ex + ey * ey;
  double s = bx - j.left_arc_x;

  // Right semicircle.  The old implementation evaluated atan2 + sin + cos for
  // every state.  For points on the admissible half-plane, radial projection
  // gives the same squared distance using one hypot; atan2 is only required if
  // this arc actually wins and its progress coordinate is needed.  Outside the
  // half-plane, the constrained projection is one of the two arc endpoints.
  double dx = px - j.right_arc_x;
  double dy = py - j.center_y;
  if (dx >= 0.0) {
    const double radial = std::hypot(dx, dy);
    const double dr = radial - j.radius;
    const double candidate_d2 = dr * dr;
    if (candidate_d2 < d2) {
      const double theta = std::atan2(dy, dx);
      d2 = candidate_d2;
      s = j.straight_length + j.radius * (theta + 0.5 * kPi);
    }
  } else {
    const bool top_endpoint = dy >= 0.0;
    ex = dx;
    ey = py - (top_endpoint ? top_y : bottom_y);
    const double candidate_d2 = ex * ex + ey * ey;
    if (candidate_d2 < d2) {
      d2 = candidate_d2;
      s = top_endpoint ? top_offset : j.straight_length;
    }
  }

  // Top straight.
  const double tx = std::min(std::max(px, j.left_arc_x), j.right_arc_x);
  ex = px - tx;
  ey = py - top_y;
  double candidate_d2 = ex * ex + ey * ey;
  if (candidate_d2 < d2) {
    d2 = candidate_d2;
    s = top_offset + (j.right_arc_x - tx);
  }

  // Left semicircle. Preserve the original angle wrapping/clamping semantics
  // exactly, including the constrained bottom endpoint for points to the right
  // of the left-arc center.  As above, only the winning radial projection needs
  // atan2 and no sin/cos are necessary.
  dx = px - j.left_arc_x;
  dy = py - j.center_y;
  const bool radial_left = dx < 0.0 || (dx == 0.0 && dy != 0.0);
  if (radial_left) {
    const double radial = std::hypot(dx, dy);
    const double dr = radial - j.radius;
    candidate_d2 = dr * dr;
    if (candidate_d2 < d2) {
      double theta = std::atan2(dy, dx);
      if (theta < 0.5 * kPi) {
        theta += 2.0 * kPi;
      }
      d2 = candidate_d2;
      s = left_offset + j.radius * (theta - 0.5 * kPi);
    }
  } else {
    ex = dx;
    ey = py - bottom_y;
    candidate_d2 = ex * ex + ey * ey;
    if (candidate_d2 < d2) {
      d2 = candidate_d2;
      s = j.track_length;
    }
  }

  // All segment formulas produce s in [0, track_length].  Avoid fmod in the
  // per-state hot path while retaining the original wrap at the final endpoint.
  if (s >= j.track_length) {
    s -= j.track_length;
  } else if (s < 0.0) {
    s += j.track_length;
  }
  *best_s = s;
  *best_d2 = d2;
}

inline void ProjectTrack(const Job& j, double xw, double yw,
                         double* best_s, double* best_d2) {
  if (j.track_kind == 1) {
    const double dx = xw - j.origin_x;
    const double dy = yw - j.origin_y;
    const double along = dx * j.rot00 + dy * j.rot01;
    const double lateral = dx * j.rot10 + dy * j.rot11;
    *best_s = along;
    *best_d2 = lateral * lateral;
    return;
  }
  ProjectStadium(j, xw, yw, best_s, best_d2);
}

}  // namespace

class FusedRolloutEvaluator {
 public:
  FusedRolloutEvaluator(
      const std::string& model_path, int nthread, int root_qpos_adr,
      int root_body_id,
      py::array_t<int, py::array::c_style | py::array::forcecast> root_geom_ids,
      int task_qpos_adr, int task_geom_id, int table_body_id,
      py::array_t<int, py::array::c_style | py::array::forcecast> table_geom_ids,
      py::array_t<int, py::array::c_style | py::array::forcecast> forbidden_box_geom_ids,
      py::array_t<int, py::array::c_style | py::array::forcecast> robot_collision_geom_ids,
      py::array_t<int, py::array::c_style | py::array::forcecast> arm_actuator_ids,
      py::array_t<int, py::array::c_style | py::array::forcecast> arm_qpos_adr,
      py::array_t<int, py::array::c_style | py::array::forcecast> arm_dof_adr,
      py::array_t<double, py::array::c_style | py::array::forcecast> joint_velocity_limits,
      py::array_t<double, py::array::c_style | py::array::forcecast> joint_acceleration_limits,
      double motion_limit_scale, double acceleration_limit_scale,
      int chunk_size)
      : nthread_(std::max(1, nthread)),
        root_qadr_(root_qpos_adr),
        task_qadr_(task_qpos_adr),
        chunk_size_(std::max(1, chunk_size)),
        root_body_id_(root_body_id),
        task_geom_id_(task_geom_id),
        table_body_id_(table_body_id),
        motion_limit_scale_(motion_limit_scale),
        acceleration_limit_scale_(acceleration_limit_scale) {
    const auto root_geoms = root_geom_ids.unchecked<1>();
    root_point_geom_ids_.reserve(static_cast<std::size_t>(root_geoms.shape(0)));
    for (py::ssize_t i = 0; i < root_geoms.shape(0); ++i) {
      root_point_geom_ids_.push_back(root_geoms(i));
    }
    const auto table_geoms = table_geom_ids.unchecked<1>();
    table_geom_ids_.reserve(static_cast<std::size_t>(table_geoms.shape(0)));
    for (py::ssize_t i = 0; i < table_geoms.shape(0); ++i) {
      table_geom_ids_.push_back(table_geoms(i));
    }
    const auto forbidden_geoms = forbidden_box_geom_ids.unchecked<1>();
    forbidden_box_geom_ids_.reserve(static_cast<std::size_t>(forbidden_geoms.shape(0)));
    for (py::ssize_t i = 0; i < forbidden_geoms.shape(0); ++i) {
      forbidden_box_geom_ids_.push_back(forbidden_geoms(i));
    }
    const auto robot_geoms = robot_collision_geom_ids.unchecked<1>();
    robot_collision_geom_ids_.reserve(static_cast<std::size_t>(robot_geoms.shape(0)));
    for (py::ssize_t i = 0; i < robot_geoms.shape(0); ++i) {
      robot_collision_geom_ids_.push_back(robot_geoms(i));
    }
    const auto arm_act = arm_actuator_ids.unchecked<1>();
    const auto arm_qpos = arm_qpos_adr.unchecked<1>();
    const auto arm_dof = arm_dof_adr.unchecked<1>();
    const auto vel_lim = joint_velocity_limits.unchecked<1>();
    const auto acc_lim = joint_acceleration_limits.unchecked<1>();
    const py::ssize_t narm = arm_act.shape(0);
    if (arm_qpos.shape(0) != narm || arm_dof.shape(0) != narm ||
        vel_lim.shape(0) != narm || acc_lim.shape(0) != narm) {
      throw std::runtime_error("arm command-limit arrays must have matching lengths");
    }
    if (!(motion_limit_scale_ > 0.0) || !std::isfinite(motion_limit_scale_) ||
        !(acceleration_limit_scale_ > 0.0) || !std::isfinite(acceleration_limit_scale_)) {
      throw std::runtime_error("motion/acceleration limit scales must be finite and positive");
    }
    arm_actuator_ids_.reserve(static_cast<std::size_t>(narm));
    arm_qpos_adr_.reserve(static_cast<std::size_t>(narm));
    arm_dof_adr_.reserve(static_cast<std::size_t>(narm));
    joint_velocity_limits_.reserve(static_cast<std::size_t>(narm));
    joint_acceleration_limits_.reserve(static_cast<std::size_t>(narm));
    for (py::ssize_t i = 0; i < narm; ++i) {
      arm_actuator_ids_.push_back(arm_act(i));
      arm_qpos_adr_.push_back(arm_qpos(i));
      arm_dof_adr_.push_back(arm_dof(i));
      joint_velocity_limits_.push_back(vel_lim(i));
      joint_acceleration_limits_.push_back(acc_lim(i));
    }
    model_ = mj_loadModel(model_path.c_str(), nullptr);
    if (!model_) {
      throw std::runtime_error("mj_loadModel failed for fused rollout model: " + model_path);
    }
    arm_actuator_mask_.assign(model_->nu, 0);
    for (const int act : arm_actuator_ids_) {
      if (act < 0 || act >= model_->nu) {
        throw std::runtime_error("invalid arm actuator id");
      }
      arm_actuator_mask_[act] = 1;
    }
    if ((root_qadr_ >= 0 && root_qadr_ + 7 > model_->nq) ||
        task_qadr_ < 0 || task_qadr_ + 7 > model_->nq ||
        root_body_id_ < 0 || root_body_id_ >= model_->nbody) {
      mj_deleteModel(model_);
      model_ = nullptr;
      throw std::runtime_error("invalid robot observation/task free-joint address for fused rollout evaluator");
    }
    for (const int geom_id : root_point_geom_ids_) {
      if (geom_id < 0 || geom_id >= model_->ngeom) {
        mj_deleteModel(model_);
        model_ = nullptr;
        throw std::runtime_error("invalid fingertip geom id for fused rollout evaluator");
      }
    }
    if (task_geom_id_ < 0 || task_geom_id_ >= model_->ngeom ||
        table_body_id_ < 0 || table_body_id_ >= model_->nbody ||
        table_geom_ids_.empty()) {
      mj_deleteModel(model_);
      model_ = nullptr;
      throw std::runtime_error("invalid task/table geom id for fused rollout evaluator");
    }
    task_half_height_ = std::max(
        1e-6, static_cast<double>(model_->geom_size[3 * task_geom_id_ + 2]));
    table_geom_mask_.assign(model_->ngeom, 0);
    for (const int geom_id : table_geom_ids_) {
      if (geom_id < 0 || geom_id >= model_->ngeom ||
          model_->geom_bodyid[geom_id] != table_body_id_) {
        mj_deleteModel(model_);
        model_ = nullptr;
        throw std::runtime_error("invalid explicit table geom id for fused rollout evaluator");
      }
      table_geom_mask_[geom_id] = 1;
    }
    for (std::size_t i = 0; i < arm_actuator_ids_.size(); ++i) {
      if (arm_actuator_ids_[i] < 0 || arm_actuator_ids_[i] >= model_->nu ||
          arm_qpos_adr_[i] < 0 || arm_qpos_adr_[i] >= model_->nq ||
          arm_dof_adr_[i] < 0 || arm_dof_adr_[i] >= model_->nv ||
          !(joint_velocity_limits_[i] > 0.0) || !std::isfinite(joint_velocity_limits_[i]) ||
          !(joint_acceleration_limits_[i] > 0.0) || !std::isfinite(joint_acceleration_limits_[i])) {
        mj_deleteModel(model_);
        model_ = nullptr;
        throw std::runtime_error("invalid Franka arm command-limit metadata");
      }
    }
    tip_geom_mask_.assign(model_->ngeom, 0);
    for (const int geom_id : root_point_geom_ids_) {
      tip_geom_mask_[geom_id] = 1;
    }
    forbidden_box_geom_mask_.assign(model_->ngeom, 0);
    for (const int geom_id : forbidden_box_geom_ids_) {
      if (geom_id < 0 || geom_id >= model_->ngeom) {
        mj_deleteModel(model_);
        model_ = nullptr;
        throw std::runtime_error("invalid forbidden box-contact geom id");
      }
      forbidden_box_geom_mask_[geom_id] = 1;
    }
    robot_geom_mask_.assign(model_->ngeom, 0);
    for (const int geom_id : robot_collision_geom_ids_) {
      if (geom_id < 0 || geom_id >= model_->ngeom) {
        mj_deleteModel(model_);
        model_ = nullptr;
        throw std::runtime_error("invalid robot collision geom id");
      }
      robot_geom_mask_[geom_id] = 1;
    }

    // Valid box contact is defined by the Python-provided allow/forbid split.
    // For the Franka task, every collision geom in the hand subtree (palm/hand
    // plus both fingers) is allowed; wrist/arm geoms remain forbidden.
    allowed_box_geom_mask_.assign(model_->ngeom, 0);
    finger_body_mask_.assign(model_->nbody, 0);
    tip_region_z_min_ = std::numeric_limits<double>::infinity();
    for (int geom_id = 0; geom_id < model_->ngeom; ++geom_id) {
      if (!robot_geom_mask_[geom_id] || forbidden_box_geom_mask_[geom_id]) {
        continue;
      }
      allowed_box_geom_mask_[geom_id] = 1;
      const int body_id = model_->geom_bodyid[geom_id];
      if (body_id >= 0 && body_id < model_->nbody) {
        finger_body_mask_[body_id] = 1;
      }
      if (model_->geom_type[geom_id] == mjGEOM_BOX) {
        const double lower = static_cast<double>(model_->geom_pos[3 * geom_id + 2]) -
                             static_cast<double>(model_->geom_size[3 * geom_id + 2]);
        tip_region_z_min_ = std::min(tip_region_z_min_, lower);
      }
    }
    if (std::isfinite(tip_region_z_min_)) {
      tip_region_z_min_ -= 0.005;
    } else {
      // Point-reaching tasks do not use task-object contact classification.
      // Keep a harmless fallback instead of requiring legacy fingertip-pad boxes.
      tip_region_z_min_ = -std::numeric_limits<double>::infinity();
    }

    nstate_ = mj_stateSize(model_, mjSTATE_FULLPHYSICS);

    // The manipulated task body is a free joint. Cache its translational DOF
    // address so the precision objective can penalize residual box velocity
    // near the target without changing the Python/C++ constructor ABI.
    task_dadr_ = -1;
    for (int j = 0; j < model_->njnt; ++j) {
      if (model_->jnt_type[j] == mjJNT_FREE &&
          model_->jnt_qposadr[j] == task_qadr_) {
        task_dadr_ = model_->jnt_dofadr[j];
        break;
      }
    }
    if (task_dadr_ < 0 || task_dadr_ + 3 > model_->nv) {
      mj_deleteModel(model_);
      model_ = nullptr;
      throw std::runtime_error("could not identify task free-joint velocity address");
    }

    // Button-task metadata.  The proxy free body identifies which button is
    // active by XY, while the actual distance target follows that button's
    // moving top site.  Press depth is read from the real slide joint.
    point_push_goal_site_id_ = mj_name2id(model_, mjOBJ_SITE, "push_goal_site");
    button_site_ids_[0] = mj_name2id(model_, mjOBJ_SITE, "button_a_top");
    button_site_ids_[1] = mj_name2id(model_, mjOBJ_SITE, "button_b_top");
    const int button_joint_ids[2] = {
        mj_name2id(model_, mjOBJ_JOINT, "button_a_joint"),
        mj_name2id(model_, mjOBJ_JOINT, "button_b_joint")};
    for (int b = 0; b < 2; ++b) {
      button_qadr_[b] = button_joint_ids[b] >= 0 ? model_->jnt_qposadr[button_joint_ids[b]] : -1;
    }
    button_tip_radius_ = !root_point_geom_ids_.empty()
        ? std::max(1e-6, static_cast<double>(model_->geom_size[3 * root_point_geom_ids_[0] + 0]))
        : 0.006;

    // Cache the torso (free-root-body) geoms and all ground/terrain geoms so
    // the fused evaluator can use the same exact fall rule as the real race:
    // inverted torso AND an actual torso-ground contact pair.
    if (root_qadr_ >= 0) {
      int free_root_body = -1;
      for (int j = 0; j < model_->njnt; ++j) {
        if (model_->jnt_type[j] == mjJNT_FREE &&
            model_->jnt_qposadr[j] == root_qadr_) {
          free_root_body = model_->jnt_bodyid[j];
          break;
        }
      }
      if (free_root_body < 0) {
        throw std::runtime_error("could not identify free-root body for fall contact test");
      }
      root_body_id_ = free_root_body;
    }
    root_geom_mask_.assign(model_->ngeom, 0);
    ground_geom_mask_.assign(model_->ngeom, 0);
    for (int g = 0; g < model_->ngeom; ++g) {
      if (model_->geom_bodyid[g] == root_body_id_) {
        root_geom_mask_[g] = 1;
      }
      bool is_ground = model_->geom_type[g] == mjGEOM_PLANE;
      const char* geom_name = mj_id2name(model_, mjOBJ_GEOM, g);
      if (geom_name) {
        const std::string name(geom_name);
        is_ground = is_ground || name == "floor" ||
                    name.rfind("race_terrain_", 0) == 0;
      }
      if (is_ground) {
        ground_geom_mask_[g] = 1;
      }
    }

    ctrl_low_.resize(model_->nu, -1.0);
    ctrl_high_.resize(model_->nu, 1.0);
    fd_epsilon_scale_.resize(model_->nu, 1.0);
    for (int k = 0; k < model_->nu; ++k) {
      if (model_->actuator_ctrllimited[k]) {
        ctrl_low_[k] = model_->actuator_ctrlrange[2 * k + 0];
        ctrl_high_[k] = model_->actuator_ctrlrange[2 * k + 1];
      }
      fd_epsilon_scale_[k] = std::max(1e-3, ctrl_high_[k] - ctrl_low_[k]);
    }
    data_.reserve(nthread_);
    for (int i = 0; i < nthread_; ++i) {
      mjData* d = mj_makeData(model_);
      if (!d) {
        CleanupData();
        mj_deleteModel(model_);
        model_ = nullptr;
        throw std::runtime_error("mj_makeData failed for fused rollout worker");
      }
      // These user-input arrays are not part of FULLPHYSICS.  They are fixed
      // at zero for this evaluator and mj_step never mutates them, so initialize
      // them once instead of clearing O(nv + nbody) memory for every rollout.
      mju_zero(d->qfrc_applied, model_->nv);
      mju_zero(d->xfrc_applied, 6 * model_->nbody);
      data_.push_back(d);
    }

    // Worker 0 is the calling thread. Keep nthread-1 persistent background
    // workers to avoid thread creation/destruction on every 20 ms control tick.
    for (int tid = 1; tid < nthread_; ++tid) {
      workers_.emplace_back([this, tid]() { WorkerLoop(tid); });
    }
  }

  void ConfigureScreening(double timestep, int iterations, int ls_iterations,
                          double tolerance) {
    if (!(timestep > 0.0) || !std::isfinite(timestep)) {
      throw std::runtime_error("screening timestep must be finite and positive");
    }
    if (!(tolerance >= 0.0) || !std::isfinite(tolerance)) {
      throw std::runtime_error("screening tolerance must be finite and nonnegative");
    }
    model_->opt.integrator = mjINT_IMPLICITFAST;
    model_->opt.timestep = timestep;
    model_->opt.iterations = std::max(1, iterations);
    model_->opt.ls_iterations = std::max(0, ls_iterations);
    model_->opt.noslip_iterations = 0;
    model_->opt.tolerance = tolerance;
  }

  ~FusedRolloutEvaluator() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    work_cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    CleanupData();
    if (model_) {
      mj_deleteModel(model_);
      model_ = nullptr;
    }
  }

  FusedRolloutEvaluator(const FusedRolloutEvaluator&) = delete;
  FusedRolloutEvaluator& operator=(const FusedRolloutEvaluator&) = delete;

  py::tuple Evaluate(
      py::array_t<double, py::array::c_style | py::array::forcecast> initial_state,
      py::array_t<double, py::array::c_style | py::array::forcecast> initial_ctrl,
      py::array_t<double, py::array::c_style | py::array::forcecast> controls,
      py::array_t<double, py::array::c_style | py::array::forcecast> nominal,
      py::array_t<double, py::array::c_style | py::array::forcecast> ctrl_scale,
      py::array_t<double, py::array::c_style | py::array::forcecast> params,
      int control_substeps) {
    const auto state_info = initial_state.request();
    const auto initial_ctrl_info = initial_ctrl.request();
    const auto ctrl_info = controls.request();
    const auto nominal_info = nominal.request();
    const auto scale_info = ctrl_scale.request();
    const auto param_info = params.request();

    if (state_info.ndim != 1 || state_info.shape[0] != nstate_) {
      throw std::runtime_error("fused initial_state has incorrect FULLPHYSICS size");
    }
    if (ctrl_info.ndim != 3) {
      throw std::runtime_error("fused controls must have shape [N,H,nu]");
    }
    const int n = static_cast<int>(ctrl_info.shape[0]);
    const int h = static_cast<int>(ctrl_info.shape[1]);
    const int nu = static_cast<int>(ctrl_info.shape[2]);
    if (n <= 0 || h <= 0 || nu != model_->nu) {
      throw std::runtime_error("fused controls dimensions do not match MuJoCo model");
    }
    if (initial_ctrl_info.ndim != 1 || initial_ctrl_info.shape[0] != nu) {
      throw std::runtime_error("fused initial_ctrl must have shape [nu]");
    }
    if (nominal_info.ndim != 2 || nominal_info.shape[0] != h ||
        nominal_info.shape[1] != nu) {
      throw std::runtime_error("fused nominal controls must have shape [H,nu]");
    }
    if (scale_info.ndim != 1 || scale_info.shape[0] != nu) {
      throw std::runtime_error("fused ctrl_scale must have shape [nu]");
    }
    if (param_info.ndim != 1 || param_info.shape[0] != 57) {
      throw std::runtime_error(
          "fused evaluator expected 57 track/cost parameters; rebuild manipulation._fused_mujoco");
    }

    py::array_t<double> positions({static_cast<py::ssize_t>(n),
                                   static_cast<py::ssize_t>(h),
                                   static_cast<py::ssize_t>(2)});
    py::array_t<double> costs({static_cast<py::ssize_t>(n)});
    py::array_t<double> progress({static_cast<py::ssize_t>(n)});
    py::array_t<bool> failed({static_cast<py::ssize_t>(n)});

    const double* p = static_cast<const double*>(param_info.ptr);
    Job job;
    job.initial_state = static_cast<const double*>(state_info.ptr);
    job.initial_ctrl = static_cast<const double*>(initial_ctrl_info.ptr);
    job.controls = static_cast<const double*>(ctrl_info.ptr);
    job.nominal = static_cast<const double*>(nominal_info.ptr);
    const double* scale = static_cast<const double*>(scale_info.ptr);
    if (ctrl_inv_scale_sq_.size() != static_cast<std::size_t>(nu)) {
      ctrl_inv_scale_sq_.resize(static_cast<std::size_t>(nu));
    }
    for (int k = 0; k < nu; ++k) {
      const double sk = std::max(std::abs(scale[k]), 1e-12);
      ctrl_inv_scale_sq_[static_cast<std::size_t>(k)] = 1.0 / (sk * sk);
    }
    job.ctrl_inv_scale_sq = ctrl_inv_scale_sq_.data();
    job.positions = positions.mutable_data();
    job.costs = costs.mutable_data();
    job.progress = progress.mutable_data();
    job.failed = failed.mutable_data();
    job.n = n;
    job.h = h;
    job.nu = nu;
    job.substeps = std::max(1, control_substeps);
    job.root_qadr = root_qadr_;
    job.root_body_id = root_body_id_;
    job.task_qadr = task_qadr_;
    job.origin_x = p[0];
    job.origin_y = p[1];
    job.rot00 = p[2];
    job.rot01 = p[3];
    job.rot10 = p[4];
    job.rot11 = p[5];
    job.canonical_start_x = p[6];
    job.canonical_start_y = p[7];
    job.radius = p[8];
    job.left_arc_x = p[9];
    job.right_arc_x = p[10];
    job.center_y = p[11];
    job.straight_length = p[12];
    job.track_length = p[13];
    job.allowed_sq = p[14];
    job.min_height = p[15];
    job.min_up = p[16];
    job.upright_weight = p[17];
    job.control_deviation_weight = p[18];
    job.box_progress_weight = p[19];
    job.robot_progress_weight = p[20];
    job.robot_box_approach_weight = p[21];
    job.box_max_lift = p[22];
    job.box_min_up = p[23];
    job.current_s = p[24];
    job.current_root_s = p[25];
    job.initial_task_root_distance = p[26];
    job.initial_task_height = p[27];
    job.start_time = p[28];
    job.track_kind = static_cast<int>(p[29]);
    job.push_standoff = p[30];
    job.goal_s = p[31];
    job.target_error_weight = p[32];
    job.terminal_target_weight = p[33];
    job.overshoot_weight = p[34];
    job.terminal_overshoot_weight = p[35];
    job.target_velocity_weight = p[36];
    job.target_tolerance = p[37];
    job.non_tip_contact_weight = p[38];
    job.tip_contact_reward = p[39];
    job.tip_push_reward = p[40];
    job.table_contact_weight = p[41];
    job.box_speed_weight = p[42];
    job.preferred_box_speed = p[43];
    job.terminal_velocity_weight = p[44];
    job.command_derivative_weight = p[45];
    job.task_mode = static_cast<int>(p[46]);
    job.box_lift_weight = p[47];
    job.reach_terminal_weight = p[48];
    job.reach_velocity_weight = p[49];
    job.reach_region_weight = p[50];
    job.reach_settle_weight = p[51];
    job.reach_velocity_ref = p[52];
    job.reach_velocity_sigma = p[53];
    job.reach_safe_radius = p[54];
    job.reach_settle_steps = static_cast<int>(p[55]);
    job.reach_acceleration_weight = p[56];
    job.task_dadr = task_dadr_;

    {
      py::gil_scoped_release release;
      RunJob(job);
    }

    return py::make_tuple(std::move(positions), std::move(costs),
                          std::move(progress), std::move(failed));
  }

  py::tuple RolloutNominal(
      py::array_t<double, py::array::c_style | py::array::forcecast> initial_state,
      py::array_t<double, py::array::c_style | py::array::forcecast> initial_ctrl,
      py::array_t<double, py::array::c_style | py::array::forcecast> controls,
      int control_substeps) {
    const auto state_info = initial_state.request();
    const auto initial_ctrl_info = initial_ctrl.request();
    const auto ctrl_info = controls.request();
    if (state_info.ndim != 1 || state_info.shape[0] != nstate_) {
      throw std::runtime_error("fused nominal initial_state has incorrect FULLPHYSICS size");
    }
    if (ctrl_info.ndim != 2 || ctrl_info.shape[0] <= 0 ||
        ctrl_info.shape[1] != model_->nu) {
      throw std::runtime_error("fused nominal controls must have shape [H,nu]");
    }
    if (initial_ctrl_info.ndim != 1 || initial_ctrl_info.shape[0] != model_->nu) {
      throw std::runtime_error("fused nominal initial_ctrl must have shape [nu]");
    }

    const int h = static_cast<int>(ctrl_info.shape[0]);
    const int nu = model_->nu;
    const int substeps = std::max(1, control_substeps);
    const double* ctrl = static_cast<const double*>(ctrl_info.ptr);
    const double* state = static_cast<const double*>(state_info.ptr);
    const double* start_ctrl = static_cast<const double*>(initial_ctrl_info.ptr);

    nominal_boundary_states_.resize(
        static_cast<std::size_t>(h + 1) * static_cast<std::size_t>(nstate_));
    nominal_warmstart_.resize(
        static_cast<std::size_t>(h + 1) * static_cast<std::size_t>(model_->nv));
    nominal_boundary_ctrl_.resize(
        static_cast<std::size_t>(h + 1) * static_cast<std::size_t>(nu));
    nominal_positions_.resize(static_cast<std::size_t>(h) * 2);
    nominal_controls_.resize(static_cast<std::size_t>(h) * static_cast<std::size_t>(nu));

    for (int t = 0; t < h; ++t) {
      for (int k = 0; k < nu; ++k) {
        const double u = ctrl[static_cast<std::size_t>(t) * nu + k];
        nominal_controls_[static_cast<std::size_t>(t) * nu + k] =
            std::min(std::max(u, ctrl_low_[k]), ctrl_high_[k]);
      }
    }

    {
      py::gil_scoped_release release;
      mjData* d = data_[0];
      mj_setState(model_, d, state, mjSTATE_FULLPHYSICS);
      for (int k = 0; k < nu; ++k) {
        d->ctrl[k] = static_cast<mjtNum>(
            std::min(std::max(start_ctrl[k], ctrl_low_[k]), ctrl_high_[k]));
      }
      mju_zero(d->qacc_warmstart, model_->nv);
      for (int w = 0; w < mjNWARNING; ++w) {
        d->warning[w].number = 0;
      }

      mj_getState(model_, d, nominal_boundary_states_.data(), mjSTATE_FULLPHYSICS);
      std::copy_n(d->qacc_warmstart, model_->nv, nominal_warmstart_.data());
      std::copy_n(d->ctrl, nu, nominal_boundary_ctrl_.data());

      bool warning_stalled = false;
      for (int t = 0; t < h; ++t) {
        const double* u = nominal_controls_.data() + static_cast<std::size_t>(t) * nu;
        ApplyRateLimitedCtrl(d, u, nu, substeps);

        if (!warning_stalled) {
          for (int sub = 0; sub < substeps; ++sub) {
            for (int w = 0; w < mjNWARNING; ++w) {
              if (d->warning[w].number) {
                warning_stalled = true;
                break;
              }
            }
            if (warning_stalled) {
              break;
            }
            mj_step(model_, d);
          }
        }

        double* boundary = nominal_boundary_states_.data()
            + static_cast<std::size_t>(t + 1) * nstate_;
        mj_getState(model_, d, boundary, mjSTATE_FULLPHYSICS);
        std::copy_n(
            d->qacc_warmstart, model_->nv,
            nominal_warmstart_.data() + static_cast<std::size_t>(t + 1) * model_->nv);
        std::copy_n(
            d->ctrl, nu,
            nominal_boundary_ctrl_.data() + static_cast<std::size_t>(t + 1) * nu);
        nominal_positions_[static_cast<std::size_t>(t) * 2 + 0] = d->qpos[task_qadr_ + 0];
        nominal_positions_[static_cast<std::size_t>(t) * 2 + 1] = d->qpos[task_qadr_ + 1];
      }
    }

    nominal_h_ = h;
    nominal_substeps_ = substeps;
    nominal_cache_valid_ = true;

    py::object base = py::cast(this, py::return_value_policy::reference);
    py::array_t<double> boundaries(
        {static_cast<py::ssize_t>(h + 1), static_cast<py::ssize_t>(nstate_)},
        {static_cast<py::ssize_t>(nstate_ * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        nominal_boundary_states_.data(), base);
    py::array_t<double> positions(
        {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(2)},
        {static_cast<py::ssize_t>(2 * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        nominal_positions_.data(), base);
    return py::make_tuple(std::move(boundaries), std::move(positions));
  }

  py::tuple EstimateSpgJacobian(
      py::array_t<double, py::array::c_style | py::array::forcecast> nominal,
      int lookahead_steps,
      double epsilon_fraction,
      int control_substeps,
      py::object time_indices) {
    const auto nominal_info = nominal.request();
    if (!nominal_cache_valid_) {
      throw std::runtime_error("fused SPG requires a preceding fused nominal rollout");
    }
    if (nominal_info.ndim != 2 || nominal_info.shape[0] != nominal_h_ ||
        nominal_info.shape[1] != model_->nu) {
      throw std::runtime_error("fused SPG nominal controls do not match cached horizon/model");
    }
    const int h = nominal_h_;
    const int nu = model_->nu;
    const int substeps = std::max(1, control_substeps);
    if (substeps != nominal_substeps_) {
      throw std::runtime_error("fused SPG substeps do not match cached nominal rollout");
    }
    if (!(epsilon_fraction >= 0.0) || !std::isfinite(epsilon_fraction)) {
      throw std::runtime_error("fused SPG epsilon_fraction must be finite and nonnegative");
    }

    const double* nominal_ptr = static_cast<const double*>(nominal_info.ptr);
    const std::size_t nctrl = static_cast<std::size_t>(h) * static_cast<std::size_t>(nu);
    for (std::size_t i = 0; i < nctrl; ++i) {
      if (nominal_ptr[i] != nominal_controls_[i]) {
        throw std::runtime_error("fused SPG nominal controls differ from cached nominal rollout");
      }
    }

    spg_ids_.clear();
    if (time_indices.is_none()) {
      spg_ids_.resize(h);
      for (int t = 0; t < h; ++t) {
        spg_ids_[t] = static_cast<std::int64_t>(t);
      }
    } else {
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ids_array(time_indices);
      const auto ids_info = ids_array.request();
      if (ids_info.ndim != 1) {
        throw std::runtime_error("fused SPG time_indices must be one-dimensional");
      }
      const auto* ids = static_cast<const std::int64_t*>(ids_info.ptr);
      for (py::ssize_t i = 0; i < ids_info.shape[0]; ++i) {
        if (ids[i] >= 0 && ids[i] < h) {
          spg_ids_.push_back(ids[i]);
        }
      }
      std::sort(spg_ids_.begin(), spg_ids_.end());
      spg_ids_.erase(std::unique(spg_ids_.begin(), spg_ids_.end()), spg_ids_.end());
    }

    const int lookahead = std::max(1, lookahead_steps);
    spg_jacobian_.resize(
        static_cast<std::size_t>(h) * 2 * static_cast<std::size_t>(nu));
    std::fill(spg_jacobian_.begin(), spg_jacobian_.end(), 0.0);
    spg_endpoints_.resize(static_cast<std::size_t>(h) * 2);
    for (int t = 0; t < h; ++t) {
      const int end_t = std::min(h - 1, t + lookahead - 1);
      spg_endpoints_[static_cast<std::size_t>(t) * 2 + 0] =
          nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 0];
      spg_endpoints_[static_cast<std::size_t>(t) * 2 + 1] =
          nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 1];
    }

    if (!spg_ids_.empty()) {
      SpgJob job;
      job.nominal = nominal_controls_.data();
      job.ids = spg_ids_.data();
      job.jacobian = spg_jacobian_.data();
      job.m = static_cast<int>(spg_ids_.size());
      job.nwork = job.m * nu;
      job.h = h;
      job.nu = nu;
      job.lookahead = lookahead;
      job.substeps = substeps;
      job.epsilon_fraction = epsilon_fraction;
      py::gil_scoped_release release;
      RunSpgJob(job);
    }

    py::object base = py::cast(this, py::return_value_policy::reference);
    py::array_t<double> jacobian(
        {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(2),
         static_cast<py::ssize_t>(nu)},
        {static_cast<py::ssize_t>(2 * nu * sizeof(double)),
         static_cast<py::ssize_t>(nu * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        spg_jacobian_.data(), base);
    py::array_t<double> endpoints(
        {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(2)},
        {static_cast<py::ssize_t>(2 * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        spg_endpoints_.data(), base);
    return py::make_tuple(std::move(jacobian), std::move(endpoints));
  }

  py::tuple EstimateSpgTimeSensitivity(
      py::array_t<double, py::array::c_style | py::array::forcecast> nominal,
      int future_steps,
      double epsilon_fraction,
      int control_substeps,
      py::object time_indices) {
    const auto nominal_info = nominal.request();
    if (!nominal_cache_valid_) {
      throw std::runtime_error("fused time-dependent SPG requires a preceding fused nominal rollout");
    }
    if (nominal_info.ndim != 2 || nominal_info.shape[0] != nominal_h_ ||
        nominal_info.shape[1] != model_->nu) {
      throw std::runtime_error("fused time-dependent SPG nominal controls do not match cached horizon/model");
    }
    const int h = nominal_h_;
    const int nu = model_->nu;
    const int substeps = std::max(1, control_substeps);
    if (substeps != nominal_substeps_) {
      throw std::runtime_error("fused time-dependent SPG substeps do not match cached nominal rollout");
    }
    if (!(epsilon_fraction >= 0.0) || !std::isfinite(epsilon_fraction)) {
      throw std::runtime_error("fused time-dependent SPG epsilon_fraction must be finite and nonnegative");
    }

    const double* nominal_ptr = static_cast<const double*>(nominal_info.ptr);
    const std::size_t nctrl = static_cast<std::size_t>(h) * static_cast<std::size_t>(nu);
    for (std::size_t i = 0; i < nctrl; ++i) {
      if (nominal_ptr[i] != nominal_controls_[i]) {
        throw std::runtime_error("fused time-dependent SPG nominal controls differ from cached nominal rollout");
      }
    }

    spg_ids_.clear();
    if (time_indices.is_none()) {
      spg_ids_.resize(h);
      for (int t = 0; t < h; ++t) {
        spg_ids_[t] = static_cast<std::int64_t>(t);
      }
    } else {
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ids_array(time_indices);
      const auto ids_info = ids_array.request();
      if (ids_info.ndim != 1) {
        throw std::runtime_error("fused time-dependent SPG time_indices must be one-dimensional");
      }
      const auto* ids = static_cast<const std::int64_t*>(ids_info.ptr);
      for (py::ssize_t i = 0; i < ids_info.shape[0]; ++i) {
        if (ids[i] >= 0 && ids[i] < h) {
          spg_ids_.push_back(ids[i]);
        }
      }
      std::sort(spg_ids_.begin(), spg_ids_.end());
      spg_ids_.erase(std::unique(spg_ids_.begin(), spg_ids_.end()), spg_ids_.end());
    }

    const int window = std::max(1, future_steps);
    spg_time_sensitivity_.resize(
        static_cast<std::size_t>(h) * static_cast<std::size_t>(window) *
        2 * static_cast<std::size_t>(nu));
    std::fill(spg_time_sensitivity_.begin(), spg_time_sensitivity_.end(), 0.0);
    spg_future_positions_.resize(
        static_cast<std::size_t>(h) * static_cast<std::size_t>(window) * 2);
    for (int t = 0; t < h; ++t) {
      for (int ell = 0; ell < window; ++ell) {
        const int end_t = std::min(h - 1, t + ell);
        const std::size_t dst = (static_cast<std::size_t>(t) * window + ell) * 2;
        spg_future_positions_[dst + 0] =
            nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 0];
        spg_future_positions_[dst + 1] =
            nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 1];
      }
    }

    if (!spg_ids_.empty()) {
      SpgJob job;
      job.nominal = nominal_controls_.data();
      job.ids = spg_ids_.data();
      job.time_sensitivity = spg_time_sensitivity_.data();
      job.m = static_cast<int>(spg_ids_.size());
      job.nwork = job.m * nu;
      job.h = h;
      job.nu = nu;
      job.lookahead = window;
      job.output_steps = window;
      job.substeps = substeps;
      job.epsilon_fraction = epsilon_fraction;
      py::gil_scoped_release release;
      RunSpgJob(job);
    }

    py::object base = py::cast(this, py::return_value_policy::reference);
    py::array_t<double> sensitivity(
        {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(window),
         static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(nu)},
        {static_cast<py::ssize_t>(window * 2 * nu * sizeof(double)),
         static_cast<py::ssize_t>(2 * nu * sizeof(double)),
         static_cast<py::ssize_t>(nu * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        spg_time_sensitivity_.data(), base);
    py::array_t<double> future_positions(
        {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(window),
         static_cast<py::ssize_t>(2)},
        {static_cast<py::ssize_t>(window * 2 * sizeof(double)),
         static_cast<py::ssize_t>(2 * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        spg_future_positions_.data(), base);
    return py::make_tuple(std::move(sensitivity), std::move(future_positions));
  }

  int nthread() const { return nthread_; }
  int nstate() const { return nstate_; }
  int chunk_size() const { return chunk_size_; }

 private:
  bool TorsoTouchingGround(const mjData* d) const {
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      const int g1 = c.geom1;
      const int g2 = c.geom2;
      if (g1 < 0 || g2 < 0 || g1 >= model_->ngeom || g2 >= model_->ngeom) {
        continue;
      }
      if ((root_geom_mask_[g1] && ground_geom_mask_[g2]) ||
          (root_geom_mask_[g2] && ground_geom_mask_[g1])) {
        return true;
      }
    }
    return false;
  }

  bool IsFingertipRegionContact(const mjData* d, const mjContact& c,
                                int robot_geom_id) const {
    if (robot_geom_id < 0 || robot_geom_id >= model_->ngeom) {
      return false;
    }
    if (allowed_box_geom_mask_[robot_geom_id]) {
      return true;
    }
    const int body_id = model_->geom_bodyid[robot_geom_id];
    if (body_id < 0 || body_id >= model_->nbody || !finger_body_mask_[body_id]) {
      return false;
    }
    const mjtNum* origin = d->xpos + 3 * body_id;
    const mjtNum* r = d->xmat + 9 * body_id;
    const double dx = static_cast<double>(c.pos[0] - origin[0]);
    const double dy = static_cast<double>(c.pos[1] - origin[1]);
    const double dz = static_cast<double>(c.pos[2] - origin[2]);
    // xmat maps local -> world; column 2 dotted with world delta is local z.
    const double local_z = static_cast<double>(r[2]) * dx +
                           static_cast<double>(r[5]) * dy +
                           static_cast<double>(r[8]) * dz;
    return local_z >= tip_region_z_min_;
  }

  void ManipulationContactFlags(const mjData* d, bool* tip_box,
                                bool* non_tip_box, bool* robot_table) const {
    *tip_box = false;
    *non_tip_box = false;
    *robot_table = false;
    if (task_geom_id_ < 0) {
      return;
    }
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      const int g1 = c.geom1;
      const int g2 = c.geom2;
      if (g1 < 0 || g2 < 0 || g1 >= model_->ngeom || g2 >= model_->ngeom) {
        continue;
      }
      int other = -1;
      if (g1 == task_geom_id_ && robot_geom_mask_[g2]) {
        other = g2;
      } else if (g2 == task_geom_id_ && robot_geom_mask_[g1]) {
        other = g1;
      }
      if (other >= 0) {
        const bool is_tip = IsFingertipRegionContact(d, c, other);
        *tip_box = *tip_box || is_tip;
        *non_tip_box = *non_tip_box || !is_tip;
      }
      const bool g1_table = table_geom_mask_[g1] != 0;
      const bool g2_table = table_geom_mask_[g2] != 0;
      if ((g1_table && robot_geom_mask_[g2]) ||
          (g2_table && robot_geom_mask_[g1])) {
        *robot_table = true;
      }
    }
  }

  double TipToTaskBoxSurfaceDistance(const mjData* d, double x, double y,
                                     double z) const {
    if (task_geom_id_ < 0) {
      return 0.0;
    }
    const mjtNum* center = d->geom_xpos + 3 * task_geom_id_;
    const double dx = x - center[0];
    const double dy = y - center[1];
    const double dz = z - center[2];
    if (model_->geom_type[task_geom_id_] != mjGEOM_BOX) {
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    // geom_xmat maps local vectors to world; transpose maps world to box-local.
    const mjtNum* r = d->geom_xmat + 9 * task_geom_id_;
    const double lx = r[0] * dx + r[3] * dy + r[6] * dz;
    const double ly = r[1] * dx + r[4] * dy + r[7] * dz;
    const double lz = r[2] * dx + r[5] * dy + r[8] * dz;
    const mjtNum* half = model_->geom_size + 3 * task_geom_id_;
    const double ox = std::max(std::abs(lx) - static_cast<double>(half[0]), 0.0);
    const double oy = std::max(std::abs(ly) - static_cast<double>(half[1]), 0.0);
    const double oz = std::max(std::abs(lz) - static_cast<double>(half[2]), 0.0);
    return std::sqrt(ox * ox + oy * oy + oz * oz);
  }

  double TipToTaskPushFaceDistance(const mjData* d, double x, double y,
                                      double z, double dir_x, double dir_y) const {
    if (task_geom_id_ < 0) {
      return 0.0;
    }
    const double dir_norm = std::hypot(dir_x, dir_y);
    if (dir_norm <= 1e-12 || model_->geom_type[task_geom_id_] != mjGEOM_BOX) {
      return TipToTaskBoxSurfaceDistance(d, x, y, z);
    }
    dir_x /= dir_norm;
    dir_y /= dir_norm;

    const mjtNum* center = d->geom_xpos + 3 * task_geom_id_;
    const double dx = x - center[0];
    const double dy = y - center[1];
    const double dz = z - center[2];
    const mjtNum* r = d->geom_xmat + 9 * task_geom_id_;
    const double local[3] = {
        r[0] * dx + r[3] * dy + r[6] * dz,
        r[1] * dx + r[4] * dy + r[7] * dz,
        r[2] * dx + r[5] * dy + r[8] * dz};
    const mjtNum* half = model_->geom_size + 3 * task_geom_id_;

    double best = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 2; ++axis) {
      // Column ``axis`` of geom_xmat is the box-local axis in world space.
      const double axis_x = r[axis];
      const double axis_y = r[3 + axis];
      for (const double sign : {-1.0, 1.0}) {
        // Inward pushing direction is opposite the outward face normal.
        const double alignment = (-sign * axis_x) * dir_x +
                                 (-sign * axis_y) * dir_y;
        if (alignment <= 1e-9) {
          continue;
        }
        double q[3] = {
            std::min(std::max(local[0], -static_cast<double>(half[0])), static_cast<double>(half[0])),
            std::min(std::max(local[1], -static_cast<double>(half[1])), static_cast<double>(half[1])),
            std::min(std::max(local[2], -static_cast<double>(half[2])), static_cast<double>(half[2]))};
        q[axis] = sign * static_cast<double>(half[axis]);
        const double ex = local[0] - q[0];
        const double ey = local[1] - q[1];
        const double ez = local[2] - q[2];
        best = std::min(best, std::sqrt(ex * ex + ey * ey + ez * ez));
      }
    }
    if (std::isfinite(best)) {
      return best;
    }
    return TipToTaskBoxSurfaceDistance(d, x, y, z);
  }

  void ApplyRateLimitedCtrl(mjData* d, const double* requested, int nu,
                            int substeps, int override_actuator = -1,
                            double override_value = 0.0) const {
    // Non-arm actuators are passed through normally. Arm actuators are
    // velocity-limited relative to the *previous commanded position* in ctrl,
    // not relative to measured q/qvel. This mirrors a motion-generator command
    // trajectory and prevents the desired position from following gravity sag.
    for (int k = 0; k < nu; ++k) {
      if (k < static_cast<int>(arm_actuator_mask_.size()) && arm_actuator_mask_[k]) {
        continue;
      }
      const double raw = (k == override_actuator) ? override_value : requested[k];
      d->ctrl[k] = static_cast<mjtNum>(
          std::min(std::max(raw, ctrl_low_[k]), ctrl_high_[k]));
    }
    if (arm_actuator_ids_.empty()) {
      return;
    }

    const double dt = std::max(1, substeps) * model_->opt.timestep;
    for (std::size_t i = 0; i < arm_actuator_ids_.size(); ++i) {
      const int act = arm_actuator_ids_[i];
      const double raw = (act == override_actuator) ? override_value : requested[act];
      const double target = std::min(std::max(raw, ctrl_low_[act]), ctrl_high_[act]);
      const double previous_command = d->ctrl[act];
      const double vmax = motion_limit_scale_ * joint_velocity_limits_[i];
      const double max_step = vmax * dt;
      const double delta = std::min(std::max(target - previous_command, -max_step), max_step);
      const double limited_target = previous_command + delta;
      d->ctrl[act] = static_cast<mjtNum>(
          std::min(std::max(limited_target, ctrl_low_[act]), ctrl_high_[act]));
    }
  }

  void CleanupData() {
    for (mjData* d : data_) {
      if (d) {
        mj_deleteData(d);
      }
    }
    data_.clear();
  }

  void RunJob(const Job& job) {
    if (nthread_ == 1) {
      current_job_ = &job;
      current_spg_job_ = nullptr;
      current_kind_ = WorkKind::kEvaluate;
      next_.store(0, std::memory_order_relaxed);
      ProcessAvailable(0);
      current_job_ = nullptr;
      current_kind_ = WorkKind::kNone;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_job_ = &job;
      current_spg_job_ = nullptr;
      current_kind_ = WorkKind::kEvaluate;
      next_.store(0, std::memory_order_relaxed);
      finished_workers_ = 0;
      ++generation_;
    }
    work_cv_.notify_all();

    // Calling thread participates as worker 0.
    ProcessAvailable(0);

    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [this]() {
        return finished_workers_ == static_cast<int>(workers_.size());
      });
      current_job_ = nullptr;
      current_kind_ = WorkKind::kNone;
    }
  }

  void RunSpgJob(const SpgJob& job) {
    if (nthread_ == 1) {
      current_job_ = nullptr;
      current_spg_job_ = &job;
      current_kind_ = WorkKind::kSpg;
      next_.store(0, std::memory_order_relaxed);
      ProcessAvailable(0);
      current_spg_job_ = nullptr;
      current_kind_ = WorkKind::kNone;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_job_ = nullptr;
      current_spg_job_ = &job;
      current_kind_ = WorkKind::kSpg;
      next_.store(0, std::memory_order_relaxed);
      finished_workers_ = 0;
      ++generation_;
    }
    work_cv_.notify_all();

    ProcessAvailable(0);

    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [this]() {
        return finished_workers_ == static_cast<int>(workers_.size());
      });
      current_spg_job_ = nullptr;
      current_kind_ = WorkKind::kNone;
    }
  }

  void WorkerLoop(int tid) {
    std::uint64_t seen_generation = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [this, &seen_generation]() {
          return stop_ || generation_ != seen_generation;
        });
        if (stop_) {
          return;
        }
        seen_generation = generation_;
      }

      ProcessAvailable(tid);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++finished_workers_;
        if (finished_workers_ == static_cast<int>(workers_.size())) {
          done_cv_.notify_one();
        }
      }
    }
  }

  void ProcessAvailable(int tid) {
    const WorkKind kind = current_kind_;
    const Job* job = current_job_;
    const SpgJob* spg_job = current_spg_job_;
    const int n = kind == WorkKind::kEvaluate
        ? (job ? job->n : 0)
        : (kind == WorkKind::kSpg && spg_job ? spg_job->nwork : 0);
    if (n <= 0) return;
    for (;;) {
      const int begin = next_.fetch_add(chunk_size_, std::memory_order_relaxed);
      if (begin >= n) {
        break;
      }
      const int end = std::min(n, begin + chunk_size_);
      for (int i = begin; i < end; ++i) {
        if (kind == WorkKind::kEvaluate) {
          EvaluateOne(*job, i, data_[tid]);
        } else if (kind == WorkKind::kSpg) {
          EvaluateSpgOne(*spg_job, i, data_[tid]);
        }
      }
    }
  }

  void EvaluateSpgOne(const SpgJob& j, int i, mjData* d) const {
    const int row = i / j.nu;
    const int actuator = i - row * j.nu;
    const int t0 = static_cast<int>(j.ids[row]);
    const int ell = std::min(j.lookahead, j.h - t0);

    const double* state = nominal_boundary_states_.data()
        + static_cast<std::size_t>(t0) * nstate_;
    mj_setState(model_, d, state, mjSTATE_FULLPHYSICS);
    std::copy_n(
        nominal_boundary_ctrl_.data() + static_cast<std::size_t>(t0) * model_->nu,
        model_->nu, d->ctrl);
    std::copy_n(
        nominal_warmstart_.data() + static_cast<std::size_t>(t0) * model_->nv,
        model_->nv, d->qacc_warmstart);
    for (int w = 0; w < mjNWARNING; ++w) {
      d->warning[w].number = 0;
    }

    const double eps = std::max(
        1e-7, j.epsilon_fraction * fd_epsilon_scale_[actuator]);
    const double u0 = j.nominal[static_cast<std::size_t>(t0) * j.nu + actuator];
    const double perturbed0 = std::min(
        std::max(u0 + eps, ctrl_low_[actuator]), ctrl_high_[actuator]);
    const double denom = perturbed0 - u0;

    bool warning_stalled = false;
    for (int r = 0; r < ell; ++r) {
      const double* u = j.nominal + static_cast<std::size_t>(t0 + r) * j.nu;
      ApplyRateLimitedCtrl(
          d, u, j.nu, j.substeps,
          r == 0 ? actuator : -1, perturbed0);
      if (!warning_stalled) {
        for (int sub = 0; sub < j.substeps; ++sub) {
          for (int w = 0; w < mjNWARNING; ++w) {
            if (d->warning[w].number) {
              warning_stalled = true;
              break;
            }
          }
          if (warning_stalled) break;
          mj_step(model_, d);
        }
      }

      if (j.time_sensitivity != nullptr && r < j.output_steps) {
        const int end_t = std::min(j.h - 1, t0 + r);
        const double base_x = nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 0];
        const double base_y = nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 1];
        const std::size_t base =
            ((static_cast<std::size_t>(t0) * j.output_steps + r) * 2) * j.nu;
        if (std::abs(denom) <= 1e-12) {
          j.time_sensitivity[base + actuator] = 0.0;
          j.time_sensitivity[base + j.nu + actuator] = 0.0;
        } else {
          j.time_sensitivity[base + actuator] =
              (d->qpos[task_qadr_ + 0] - base_x) / denom;
          j.time_sensitivity[base + j.nu + actuator] =
              (d->qpos[task_qadr_ + 1] - base_y) / denom;
        }
      }
    }

    if (j.jacobian == nullptr) {
      return;
    }
    double* out = j.jacobian + static_cast<std::size_t>(t0) * 2 * j.nu;
    if (std::abs(denom) <= 1e-12) {
      out[actuator] = 0.0;
      out[j.nu + actuator] = 0.0;
      return;
    }
    const int end_t = std::min(j.h - 1, t0 + j.lookahead - 1);
    const double base_x = nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 0];
    const double base_y = nominal_positions_[static_cast<std::size_t>(end_t) * 2 + 1];
    out[actuator] = (d->qpos[task_qadr_ + 0] - base_x) / denom;
    out[j.nu + actuator] = (d->qpos[task_qadr_ + 1] - base_y) / denom;
  }

  int ActiveButtonIndex(const mjData* d) const {
    if (button_site_ids_[0] < 0 || button_site_ids_[1] < 0) return -1;
    const double tx = d->qpos[task_qadr_ + 0];
    const double ty = d->qpos[task_qadr_ + 1];
    double best = std::numeric_limits<double>::infinity();
    int best_idx = -1;
    for (int b = 0; b < 2; ++b) {
      const int sid = button_site_ids_[b];
      const double dx = d->site_xpos[3 * sid + 0] - tx;
      const double dy = d->site_xpos[3 * sid + 1] - ty;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best) { best = d2; best_idx = b; }
    }
    return best_idx;
  }

  void EvaluateOne(const Job& j, int i, mjData* d) const {
    // Shared simulator for the existing tasks plus Franka box point-pushing.
    // Task mode 4 uses only tool-box surface, box-goal and control-deviation
    // distances; all other costs remain unchanged for earlier task modes.
    mj_setState(model_, d, j.initial_state, mjSTATE_FULLPHYSICS);
    for (int k = 0; k < j.nu; ++k) {
      d->ctrl[k] = static_cast<mjtNum>(
          std::min(std::max(j.initial_ctrl[k], ctrl_low_[k]), ctrl_high_[k]));
    }
    mju_zero(d->qacc_warmstart, model_->nv);
    for (int w = 0; w < mjNWARNING; ++w) {
      d->warning[w].number = 0;
    }

    const double* controls_i = j.controls + static_cast<std::size_t>(i) * j.h * j.nu;
    double* positions_i = j.positions + static_cast<std::size_t>(i) * j.h * 2;

    double s_prev = j.current_s;
    double root_s_prev = j.current_root_s;
    double cumulative = 0.0;
    double root_cumulative = 0.0;
    double prefix_sum = 0.0;
    double root_prefix_sum = 0.0;
    double face_distance_sum = 0.0;
    double point_push_goal_distance_sum = 0.0;
    double reach_cost = 0.0;
    double settle_cost = 0.0;
    double last_distance = 0.0;
    double last_speed_sq = 0.0;
    double previous_v[3] = {0, 0, 0};
    bool have_previous_v = false;
    double non_tip_contact_fraction_sum = 0.0;
    double table_contact_fraction_sum = 0.0;
    double box_lift_sq_sum = 0.0;
    double upright_cost = 0.0;
    double press_error_sq_sum = 0.0;
    double control_cost = 0.0;
    double acceleration_cost = 0.0;
    double prev_time = j.start_time;
    bool failed = false;
    int last_t = -1;

    const double half_track = 0.5 * j.track_length;
    for (int t = 0; t < j.h; ++t) {
      const double* u = controls_i + static_cast<std::size_t>(t) * j.nu;
      ApplyRateLimitedCtrl(d, u, j.nu, j.substeps);

      bool warning_stalled = false;
      int bad_contact_substeps = 0;
      int table_contact_substeps = 0;
      for (int sub = 0; sub < j.substeps; ++sub) {
        for (int w = 0; w < mjNWARNING; ++w) {
          if (d->warning[w].number) {
            warning_stalled = true;
            break;
          }
        }
        if (warning_stalled) {
          break;
        }
        mj_step(model_, d);
        bool tip_box = false;
        bool non_tip_box = false;
        bool robot_table = false;
        ManipulationContactFlags(d, &tip_box, &non_tip_box, &robot_table);
        if (non_tip_box) {
          ++bad_contact_substeps;
        }
        if (robot_table) {
          ++table_contact_substeps;
        }
      }
      non_tip_contact_fraction_sum +=
          static_cast<double>(bad_contact_substeps) /
          static_cast<double>(std::max(1, j.substeps));
      table_contact_fraction_sum +=
          static_cast<double>(table_contact_substeps) /
          static_cast<double>(std::max(1, j.substeps));

      const double xw = d->qpos[j.task_qadr + 0];
      const double yw = d->qpos[j.task_qadr + 1];
      double root_x = xw;
      double root_y = yw;
      double root_z = d->qpos[j.task_qadr + 2];
      double up = 1.0;
      if (j.root_qadr >= 0) {
        root_x = d->qpos[j.root_qadr + 0];
        root_y = d->qpos[j.root_qadr + 1];
        root_z = d->qpos[j.root_qadr + 2];
        const double qx = d->qpos[j.root_qadr + 4];
        const double qy = d->qpos[j.root_qadr + 5];
        up = 1.0 - 2.0 * (qx * qx + qy * qy);
      } else if (!root_point_geom_ids_.empty()) {
        root_x = 0.0;
        root_y = 0.0;
        root_z = 0.0;
        for (const int geom_id : root_point_geom_ids_) {
          root_x += d->geom_xpos[3 * geom_id + 0];
          root_y += d->geom_xpos[3 * geom_id + 1];
          root_z += d->geom_xpos[3 * geom_id + 2];
        }
        const double inv_ngeom = 1.0 / static_cast<double>(root_point_geom_ids_.size());
        root_x *= inv_ngeom;
        root_y *= inv_ngeom;
        root_z *= inv_ngeom;
        // For fixed-base manipulation tasks, interpret `up` as downward
        // alignment of the tool body's local +Z axis with world -Z.
        up = -d->xmat[9 * j.root_body_id + 8];
      } else {
        root_x = d->xpos[3 * j.root_body_id + 0];
        root_y = d->xpos[3 * j.root_body_id + 1];
        root_z = d->xpos[3 * j.root_body_id + 2];
      }

      positions_i[2 * t + 0] = xw;
      positions_i[2 * t + 1] = yw;
      last_t = t;

      double best_s = 0.0;
      double best_d2 = 0.0;
      ProjectTrack(j, xw, yw, &best_s, &best_d2);
      double root_s = best_s;
      double root_d2 = best_d2;
      if (j.task_qadr != j.root_qadr) {
        ProjectTrack(j, root_x, root_y, &root_s, &root_d2);
      }

      if (j.task_qadr != j.root_qadr && j.task_mode != 4) {
        const double task_z = d->qpos[j.task_qadr + 2];
        const double task_qx = d->qpos[j.task_qadr + 4];
        const double task_qy = d->qpos[j.task_qadr + 5];
        const double task_up = 1.0 - 2.0 * (task_qx * task_qx + task_qy * task_qy);
        if (task_z > j.initial_task_height + j.box_max_lift ||
            task_up < j.box_min_up) {
          failed = true;
          break;
        }
        const double lift = std::max(0.0, task_z - j.initial_task_height);
        const double lift_norm = lift / task_half_height_;
        box_lift_sq_sum += lift_norm * lift_norm;
      }

      const bool fell = j.root_qadr >= 0 && (up < j.min_up) && TorsoTouchingGround(d);
      if ((j.task_mode != 4 && (best_d2 > j.allowed_sq || root_d2 > j.allowed_sq)) ||
          fell || d->time <= prev_time + 1e-15) {
        failed = true;
        break;
      }

      double ds = best_s - s_prev;
      if (j.track_kind == 0) {
        if (ds > half_track) ds -= j.track_length;
        else if (ds < -half_track) ds += j.track_length;
      }
      cumulative += ds;
      prefix_sum += cumulative;
      s_prev = best_s;

      if (j.task_qadr != j.root_qadr) {
        root_s_prev = root_s;
        if (j.task_mode == 2) {
          const int active_button = ActiveButtonIndex(d);
          if (active_button < 0 || button_qadr_[active_button] < 0) {
            failed = true;
            break;
          }
          const int sid = button_site_ids_[active_button];
          const double tx = d->site_xpos[3 * sid + 0];
          const double ty = d->site_xpos[3 * sid + 1];
          const double tz = d->site_xpos[3 * sid + 2] + button_tip_radius_;
          const double ex = root_x - tx;
          const double ey = root_y - ty;
          const double ez = root_z - tz;
          face_distance_sum += std::sqrt(ex * ex + ey * ey + ez * ez);

          const double desired_press = std::max(j.target_tolerance, 1e-9);
          const double press_depth = std::max(0.0, -static_cast<double>(d->qpos[button_qadr_[active_button]]));
          const double press_err = std::max(0.0, desired_press - press_depth) / desired_press;
          press_error_sq_sum += press_err * press_err;
        } else if (j.task_mode == 4) {
          // Point-to-point box pushing: no orientation, progress, contact or lift
          // penalties. The approach term is tip-to-box *surface* distance.
          if (point_push_goal_site_id_ < 0) {
            failed = true;
            break;
          }
          face_distance_sum += TipToTaskBoxSurfaceDistance(d, root_x, root_y, root_z);
          const mjtNum* goal = d->site_xpos + 3 * point_push_goal_site_id_;
          const double gx = xw - static_cast<double>(goal[0]);
          const double gy = yw - static_cast<double>(goal[1]);
          const double gz = d->qpos[j.task_qadr + 2] - static_cast<double>(goal[2]);
          point_push_goal_distance_sum += std::sqrt(gx * gx + gy * gy + gz * gz);
        } else if (j.task_mode == 3) {
          // Clean Franka benchmark: direct 3-D distance from the controlled
          // fingertip point to a fixed free-body target.  No object/contact
          // manipulation semantics are involved in this mode.
          const double tx = d->qpos[j.task_qadr + 0];
          const double ty = d->qpos[j.task_qadr + 1];
          const double tz = d->qpos[j.task_qadr + 2];
          const double ex = root_x - tx;
          const double ey = root_y - ty;
          const double ez = root_z - tz;
          const double distance = std::sqrt(ex * ex + ey * ey + ez * ez);
          face_distance_sum += distance;
          last_distance = distance;
          const double radius = std::max(j.reach_safe_radius, 1e-9);
          const double outside = std::max(0.0, distance / radius - 1.0);
          reach_cost += j.reach_region_weight * outside * outside;
          // Finite-difference Cartesian velocity at the actual simulated tip,
          // not at the commanded joint positions. First sample is unpenalized.
          const double current_tip[3] = {root_x, root_y, root_z};
          double v_sq = 0.0;
          if (have_previous_v) {
            const double dt = std::max(d->time - prev_time, 1e-9);
            for (int axis = 0; axis < 3; ++axis) {
              const double v = (current_tip[axis] - previous_v[axis]) / dt;
              v_sq += v * v;
            }
            const double normalized_v2 = v_sq / std::max(j.reach_velocity_ref * j.reach_velocity_ref, 1e-12);
            const double gate = std::exp(-distance * distance /
                                     std::max(2.0 * j.reach_velocity_sigma * j.reach_velocity_sigma, 1e-12));
            reach_cost += j.reach_velocity_weight * gate * normalized_v2;
          }
          for (int axis = 0; axis < 3; ++axis) previous_v[axis] = current_tip[axis];
          have_previous_v = true;
          last_speed_sq = v_sq;
          if (t >= j.h - j.reach_settle_steps) {
            const double normalized_v2 = v_sq / std::max(j.reach_velocity_ref * j.reach_velocity_ref, 1e-12);
            settle_cost += (distance * distance / (radius * radius)) + normalized_v2;
          }
        } else if (j.track_kind == 1) {
          face_distance_sum += TipToTaskPushFaceDistance(
              d, root_x, root_y, root_z, j.rot00, j.rot01);
        } else {
          face_distance_sum += TipToTaskBoxSurfaceDistance(d, root_x, root_y, root_z);
        }
      }

      prev_time = d->time;

      double du_sq_sum = 0.0;
      const double* unom = j.nominal + static_cast<std::size_t>(t) * j.nu;
      for (int k = 0; k < j.nu; ++k) {
        const double diff = u[k] - unom[k];
        du_sq_sum += diff * diff * j.ctrl_inv_scale_sq[k];
      }
      control_cost += j.control_deviation_weight * (du_sq_sum / j.nu);
      if (j.task_mode == 3 && t >= 2 && j.reach_acceleration_weight > 0.0) {
        const double* uprev = controls_i + static_cast<std::size_t>(t - 1) * j.nu;
        const double* uprev2 = controls_i + static_cast<std::size_t>(t - 2) * j.nu;
        double curvature = 0.0;
        for (int k = 0; k < j.nu; ++k) {
          const double a = u[k] - 2.0 * uprev[k] + uprev2[k];
          curvature += a * a * j.ctrl_inv_scale_sq[k];
        }
        acceleration_cost += curvature / std::max(1, j.nu);
      }
      const double err_up = 1.0 - up;
      upright_cost += j.upright_weight * err_up * err_up;
    }

    if (failed && last_t >= 0) {
      const double x = positions_i[2 * last_t + 0];
      const double y = positions_i[2 * last_t + 1];
      for (int t = last_t + 1; t < j.h; ++t) {
        positions_i[2 * t + 0] = x;
        positions_i[2 * t + 1] = y;
      }
    }

    j.progress[i] = cumulative;
    j.failed[i] = failed;
    if (failed) {
      j.costs[i] = std::numeric_limits<double>::infinity();
    } else {
      const double inv_h = 1.0 / static_cast<double>(std::max(1, j.h));
      const double face_scale = std::max(j.initial_task_root_distance, 0.02);
      double face_cost = 0.0;
      if (j.task_qadr != j.root_qadr) {
        face_cost = j.robot_box_approach_weight *
                    face_distance_sum * inv_h / face_scale;
      }
      const double contact_cost =
          j.non_tip_contact_weight * non_tip_contact_fraction_sum * inv_h;
      const double table_fraction = table_contact_fraction_sum * inv_h;
      const double table_cost =
          j.table_contact_weight * table_fraction * table_fraction;
      const double lift_cost = j.box_lift_weight * box_lift_sq_sum * inv_h;
      const double orientation_cost = upright_cost * inv_h;
      const double press_cost = j.target_error_weight * press_error_sq_sum * inv_h;
      if (j.task_mode == 3) {
        const double settle_count = std::max(1, std::min(j.h, j.reach_settle_steps));
        reach_cost = reach_cost * inv_h + j.reach_settle_weight * settle_cost / settle_count
                     + j.reach_terminal_weight * last_distance / face_scale;
      }
      if (j.task_mode == 4) {
        // Three-term objective only: both distances averaged over the horizon
        // plus the same tiny nominal command regularizer. No extra shaping.
        j.costs[i] = j.robot_box_approach_weight * face_distance_sum * inv_h
                     + j.target_error_weight * point_push_goal_distance_sum * inv_h
                     + control_cost * inv_h;
      } else {
        j.costs[i] = face_cost + press_cost + orientation_cost + contact_cost + table_cost + lift_cost + control_cost + reach_cost + j.reach_acceleration_weight * acceleration_cost * inv_h;
      }
    }
  }

  mjModel* model_ = nullptr;
  std::vector<mjData*> data_;
  int nthread_ = 1;
  int root_qadr_ = -1;
  int task_qadr_ = 0;
  int chunk_size_ = 1;
  int nstate_ = 0;
  int root_body_id_ = -1;
  std::vector<int> root_point_geom_ids_;
  int task_geom_id_ = -1;
  int table_body_id_ = -1;
  std::vector<int> table_geom_ids_;
  std::vector<unsigned char> table_geom_mask_;
  double task_half_height_ = 0.05;
  std::vector<int> forbidden_box_geom_ids_;
  std::vector<int> robot_collision_geom_ids_;
  std::vector<int> arm_actuator_ids_;
  std::vector<unsigned char> arm_actuator_mask_;
  std::vector<int> arm_qpos_adr_;
  std::vector<int> arm_dof_adr_;
  std::vector<double> joint_velocity_limits_;
  std::vector<double> joint_acceleration_limits_;
  double motion_limit_scale_ = 1.0;
  double acceleration_limit_scale_ = 1.0;
  std::vector<unsigned char> tip_geom_mask_;
  std::vector<unsigned char> forbidden_box_geom_mask_;
  std::vector<unsigned char> robot_geom_mask_;
  std::vector<unsigned char> allowed_box_geom_mask_;
  std::vector<unsigned char> finger_body_mask_;
  double tip_region_z_min_ = 0.0;
  int task_dadr_ = -1;
  int point_push_goal_site_id_ = -1;
  int button_site_ids_[2] = {-1, -1};
  int button_qadr_[2] = {-1, -1};
  double button_tip_radius_ = 0.006;
  std::vector<unsigned char> root_geom_mask_;
  std::vector<unsigned char> ground_geom_mask_;
  std::vector<double> ctrl_low_;
  std::vector<double> ctrl_high_;
  std::vector<double> fd_epsilon_scale_;
  std::vector<double> ctrl_inv_scale_sq_;

  bool nominal_cache_valid_ = false;
  int nominal_h_ = 0;
  int nominal_substeps_ = 1;
  std::vector<double> nominal_boundary_states_;
  std::vector<double> nominal_warmstart_;
  std::vector<double> nominal_boundary_ctrl_;
  std::vector<double> nominal_positions_;
  std::vector<double> nominal_controls_;
  std::vector<std::int64_t> spg_ids_;
  std::vector<double> spg_jacobian_;
  std::vector<double> spg_endpoints_;
  std::vector<double> spg_time_sensitivity_;
  std::vector<double> spg_future_positions_;

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  bool stop_ = false;
  std::uint64_t generation_ = 0;
  int finished_workers_ = 0;
  const Job* current_job_ = nullptr;
  const SpgJob* current_spg_job_ = nullptr;
  WorkKind current_kind_ = WorkKind::kNone;
  std::atomic<int> next_{0};
};

PYBIND11_MODULE(_fused_mujoco, m) {
  m.doc() = "Fused MuJoCo rollout + track/task cost evaluator";
  py::class_<FusedRolloutEvaluator>(m, "FusedRolloutEvaluator")
      .def(py::init<
               const std::string&, int, int, int,
               py::array_t<int, py::array::c_style | py::array::forcecast>,
               int, int, int,
               py::array_t<int, py::array::c_style | py::array::forcecast>,
               py::array_t<int, py::array::c_style | py::array::forcecast>,
               py::array_t<int, py::array::c_style | py::array::forcecast>,
               py::array_t<int, py::array::c_style | py::array::forcecast>,
               py::array_t<int, py::array::c_style | py::array::forcecast>,
               py::array_t<int, py::array::c_style | py::array::forcecast>,
               py::array_t<double, py::array::c_style | py::array::forcecast>,
               py::array_t<double, py::array::c_style | py::array::forcecast>,
               double, double,
               int>(),
           py::arg("model_path"), py::arg("nthread"),
           py::arg("root_qpos_adr"), py::arg("root_body_id"),
           py::arg("root_geom_ids"),
           py::arg("task_qpos_adr"), py::arg("task_geom_id"), py::arg("table_body_id"),
           py::arg("table_geom_ids"),
           py::arg("forbidden_box_geom_ids"), py::arg("robot_collision_geom_ids"),
           py::arg("arm_actuator_ids"), py::arg("arm_qpos_adr"), py::arg("arm_dof_adr"),
           py::arg("joint_velocity_limits"), py::arg("joint_acceleration_limits"),
           py::arg("motion_limit_scale"), py::arg("acceleration_limit_scale"),
           py::arg("chunk_size") = 1)
      .def("evaluate", &FusedRolloutEvaluator::Evaluate,
           py::arg("initial_state"), py::arg("initial_ctrl"), py::arg("controls"),
           py::arg("nominal_controls"), py::arg("ctrl_scale"),
           py::arg("params"), py::arg("control_substeps"))
      .def("configure_screening", &FusedRolloutEvaluator::ConfigureScreening,
           py::arg("timestep"), py::arg("iterations") = 5,
           py::arg("ls_iterations") = 1, py::arg("tolerance") = 1e-4)
      .def("rollout_nominal", &FusedRolloutEvaluator::RolloutNominal,
           py::arg("initial_state"), py::arg("initial_ctrl"),
           py::arg("controls"), py::arg("control_substeps"))
      .def("estimate_spg_jacobian", &FusedRolloutEvaluator::EstimateSpgJacobian,
           py::arg("nominal_controls"), py::arg("lookahead_steps"),
           py::arg("epsilon_fraction"), py::arg("control_substeps"),
           py::arg("time_indices") = py::none())
      .def("estimate_spg_time_sensitivity",
           &FusedRolloutEvaluator::EstimateSpgTimeSensitivity,
           py::arg("nominal_controls"), py::arg("future_steps"),
           py::arg("epsilon_fraction"), py::arg("control_substeps"),
           py::arg("time_indices") = py::none())
      .def_property_readonly("nthread", &FusedRolloutEvaluator::nthread)
      .def_property_readonly("nstate", &FusedRolloutEvaluator::nstate)
      .def_property_readonly("chunk_size", &FusedRolloutEvaluator::chunk_size)
      .def_property_readonly("cost_param_count", [](const FusedRolloutEvaluator&) { return 57; })
      .def_property_readonly("root_point_abi", [](const FusedRolloutEvaluator&) { return 4; })
      .def_property_readonly("box_contact_policy_abi", [](const FusedRolloutEvaluator&) { return 6; })
      .def_property_readonly("table_contact_policy_abi", [](const FusedRolloutEvaluator&) { return 1; })
      .def_property_readonly("command_limit_abi", [](const FusedRolloutEvaluator&) { return 2; })
      .def_property_readonly("simple_reward_abi", [](const FusedRolloutEvaluator&) { return 5; })
      .def_property_readonly("point_push_abi", [](const FusedRolloutEvaluator&) { return 1; });
}
