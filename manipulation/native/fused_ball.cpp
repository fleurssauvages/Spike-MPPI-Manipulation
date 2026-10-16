// Contact-aware, cost-fused MuJoCo rollouts for Franka ball balancing/bouncing.
// This is a separate evaluator from the original reaching/box-pushing code.
// The fidelity of the planner is the same as the executed plant: both load the
// same mjModel and use the same mj_step integrator/contact/fluid settings.
#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
using CArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline double sq(double x) { return x*x; }
inline double clip(double x, double low, double high) {
  return std::min(std::max(x, low), high);
}
inline double dot3(const double* x, const double* y) {
  return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

// Weights have identical order to dataclasses.astuple(BalanceWeights/BounceWeights).
struct BallJob {
  const double *state = nullptr, *initial_ctrl = nullptr, *controls = nullptr;
  const double *nominal = nullptr, *scale = nullptr, *weights = nullptr;
  double *positions = nullptr, *costs = nullptr, *progress = nullptr;
  bool *failed = nullptr;
  int n = 0, h = 0, nu = 0, substeps = 1;
};

class BallFusedEvaluator {
 public:
  BallFusedEvaluator(const std::string& model_path, int nthread, int chunk_size,
                     const std::string& task, CArray velocity_limits,
                     double motion_limit_scale)
      : nthread_(std::max(1, nthread)), chunk_(std::max(1, chunk_size)),
        bouncing_(task == "bouncing"), motion_scale_(motion_limit_scale) {
    if (task != "bouncing" && task != "balancing") {
      throw std::runtime_error("BallFusedEvaluator task must be balancing or bouncing");
    }
    if (!(motion_scale_ > 0.0) || !std::isfinite(motion_scale_)) {
      throw std::runtime_error("invalid motion_limit_scale");
    }
    model_ = mj_loadModel(model_path.c_str(), nullptr);
    if (!model_) throw std::runtime_error("cannot load native ball MuJoCo model");
    if (model_->nu <= 0 || velocity_limits.ndim() != 1 ||
        velocity_limits.shape(0) != model_->nu) {
      throw std::runtime_error("ball velocity limits must have one entry per actuator");
    }
    for (int k = 0; k < model_->nu; ++k) {
      const double v = velocity_limits.data()[k];
      if (!(v > 0.0) || !std::isfinite(v)) {
        throw std::runtime_error("invalid ball robot joint velocity limit");
      }
      vmax_.push_back(v);
      const bool limited = model_->actuator_ctrllimited[k];
      low_.push_back(limited ? model_->actuator_ctrlrange[2*k] : -1e6);
      high_.push_back(limited ? model_->actuator_ctrlrange[2*k+1] : 1e6);
    }
    const int ball_geom = mj_name2id(model_, mjOBJ_GEOM, "reach_target_geom");
    const int tool_geom = mj_name2id(model_, mjOBJ_GEOM, "ball_tool_geom");
    if (ball_geom < 0 || tool_geom < 0) {
      throw std::runtime_error("native ball evaluator: missing ball/racket geom");
    }
    radius_ = model_->geom_size[3*ball_geom];
    usable_ = std::max(1e-3, model_->geom_size[3*tool_geom] - radius_);
    const int free_joint = mj_name2id(model_, mjOBJ_JOINT, "reach_target_free");
    if (free_joint < 0 || model_->jnt_type[free_joint] != mjJNT_FREE) {
      throw std::runtime_error("native ball evaluator: missing ball free joint");
    }
    qadr_ = model_->jnt_qposadr[free_joint];
    dadr_ = model_->jnt_dofadr[free_joint];
    const char* sensor_names[6] = {
      "ball_tool_pos", "ball_tool_x", "ball_tool_y", "ball_tool_z",
      "ball_tool_linvel", "ball_tool_angvel"
    };
    for (int k=0; k<6; ++k) {
      const int id=mj_name2id(model_, mjOBJ_SENSOR, sensor_names[k]);
      if (id<0 || model_->sensor_dim[id]!=3)
        throw std::runtime_error(std::string("native ball evaluator: sensor missing: ") + sensor_names[k]);
      sensor_adr_[k]=model_->sensor_adr[id];
    }
    state_size_ = mj_stateSize(model_, mjSTATE_FULLPHYSICS);
    data_.reserve(nthread_);
    for (int t=0; t<nthread_; ++t) {
      mjData* d = mj_makeData(model_);
      if (!d) throw std::runtime_error("mj_makeData failed for ball worker");
      mju_zero(d->qfrc_applied, model_->nv);
      mju_zero(d->xfrc_applied, 6*model_->nbody);
      data_.push_back(d);
    }
    for (int t=1; t<nthread_; ++t) {
      threads_.emplace_back([this,t]{ worker(t); });
    }
  }
  ~BallFusedEvaluator() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (auto& th: threads_) if (th.joinable()) th.join();
    for (mjData* d: data_) mj_deleteData(d);
    if (model_) mj_deleteModel(model_);
  }
  BallFusedEvaluator(const BallFusedEvaluator&) = delete;
  BallFusedEvaluator& operator=(const BallFusedEvaluator&) = delete;

  py::tuple Evaluate(CArray state, CArray initial_ctrl, CArray controls,
                     CArray nominal, CArray ctrl_scale, CArray weights,
                     int substeps) {
    const int nu=model_->nu;
    if (state.ndim()!=1 || state.shape(0)!=state_size_ ||
        initial_ctrl.ndim()!=1 || initial_ctrl.shape(0)!=nu ||
        controls.ndim()!=3 || controls.shape(0)<1 || controls.shape(1)<1 ||
        controls.shape(2)!=nu || nominal.ndim()!=2 ||
        nominal.shape(0)!=controls.shape(1) || nominal.shape(1)!=nu ||
        ctrl_scale.ndim()!=1 || ctrl_scale.shape(0)!=nu ||
        weights.ndim()!=1 || weights.shape(0)!=(bouncing_?12:8) ||
        substeps<1) {
      throw std::runtime_error("native ball evaluator: invalid state, control, or weights shape");
    }
    const int n=static_cast<int>(controls.shape(0));
    const int h=static_cast<int>(controls.shape(1));
    py::array_t<double> positions({static_cast<py::ssize_t>(n),
                                   static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(2)});
    py::array_t<double> costs({static_cast<py::ssize_t>(n)});
    py::array_t<double> progress({static_cast<py::ssize_t>(n)});
    py::array_t<bool> failed({static_cast<py::ssize_t>(n)});
    BallJob job{state.data(), initial_ctrl.data(), controls.data(), nominal.data(),
                ctrl_scale.data(), weights.data(), positions.mutable_data(),
                costs.mutable_data(), progress.mutable_data(), failed.mutable_data(),
                n,h,nu,substeps};
    {
      py::gil_scoped_release release;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        job_=&job;
        next_.store(0);
        finished_=0;
        ++generation_;
      }
      cv_.notify_all();
      process(job, 0);
      std::unique_lock<std::mutex> lock(mutex_);
      done_.wait(lock, [this]{return finished_==nthread_-1;});
      job_=nullptr;
    }
    return py::make_tuple(std::move(positions), std::move(costs),
                          std::move(progress), std::move(failed));
  }
  int nthread() const { return nthread_; }
  int nstate() const { return state_size_; }
  int chunk_size() const { return chunk_; }
  int task_abi() const { return 1; }

 private:
  void worker(int tid) {
    std::uint64_t seen=0;
    for (;;) {
      const BallJob* job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this,&seen]{return stop_ || generation_!=seen;});
        if (stop_) return;
        seen=generation_;
        job=job_;
      }
      if (job) process(*job, tid);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++finished_;
        if (finished_==nthread_-1) done_.notify_one();
      }
    }
  }
  void process(const BallJob& job, int tid) {
    for (;;) {
      const int first=next_.fetch_add(chunk_, std::memory_order_relaxed);
      if (first>=job.n) break;
      const int last=std::min(job.n, first+chunk_);
      for (int i=first; i<last; ++i) one(job,i,data_[tid]);
    }
  }

  void one(const BallJob& job, int i, mjData* d) const {
    mj_setState(model_,d,job.state,mjSTATE_FULLPHYSICS);
    for (int k=0;k<job.nu;++k) d->ctrl[k]=clip(job.initial_ctrl[k], low_[k],high_[k]);
    mju_zero(d->qacc_warmstart, model_->nv);
    for (int w=0;w<mjNWARNING;++w) d->warning[w].number=0;

    const int h=job.h, substeps=job.substeps, nu=job.nu;
    const int frames=h*substeps;
    const double *candidate=job.controls+static_cast<std::size_t>(i)*h*nu;
    double *xyout=job.positions+static_cast<std::size_t>(i)*h*2;
    const double *w=job.weights;
    std::vector<double> previous(nu), initial(nu), actual(nu);
    for (int k=0;k<nu;++k) previous[k]=initial[k]=d->ctrl[k];
    double du_sum=0.0, proposal_sum=0.0;
    double position_sum=0.0, velocity_sum=0.0, tilt_sum=0.0;
    double gap_sum=0.0, lost_sum=0.0, last_position=0, last_velocity=0;
    bool any_lost=false;
    double landing_sum=0.0, last_landing=0.0, preimpact_sum=0.0;
    double predicted_sum=0.0, impact_sum=0.0, apex_sum=0.0, work_sum=0.0;
    int n_hit=0, n_ascent=0;
    bool prev_hit=false, have_vz=false;
    double prev_vz=0.0, first_tool_x=0.0, first_tool_y=0.0;
    const double desired_vz=bouncing_ ? std::sqrt(2.0*9.81*w[10]) : 0.0;
    const double e=bouncing_ ? w[11] : 0.0;
    bool diverged=false;
    int tdone=0;
    for(int t=0;t<h&&!diverged;++t) {
      double delta=0.0, deviation=0.0;
      for (int k=0;k<nu;++k) {
        const int j=t*nu+k;
        const double goal=clip(candidate[j],low_[k],high_[k]);
        const double step=vmax_[k]*motion_scale_*substeps*model_->opt.timestep;
        const double u=clip(previous[k]+clip(goal-previous[k],-step,step),low_[k],high_[k]);
        actual[k]=u;
        const double inv=1.0/std::max(job.scale[k],1e-6);
        delta+=sq((u-previous[k])*inv);
        deviation+=sq((u-job.nominal[j])*inv);
        d->ctrl[k]=u;
      }
      du_sum+=delta/nu;
      proposal_sum+=deviation/nu;
      for(int k=0;k<nu;++k) previous[k]=actual[k];
      for (int sub=0;sub<substeps;++sub) {
        mj_step(model_,d);
        for(int wi=0;wi<mjNWARNING;++wi) {
          if(d->warning[wi].number!=0) { diverged=true; break; }
        }
        if(diverged) break;
        const double *p = d->qpos+qadr_;
        const double *v = d->qvel+dadr_;
        const double *s = d->sensordata;
        const double *tip=s+sensor_adr_[0];
        const double *ax=s+sensor_adr_[1];
        const double *ay=s+sensor_adr_[2];
        const double *az=s+sensor_adr_[3];
        const double *lin=s+sensor_adr_[4];
        const double *ang=s+sensor_adr_[5];
        const double dr[3]={p[0]-tip[0],p[1]-tip[1],p[2]-tip[2]};
        const double cross[3]={ang[1]*dr[2]-ang[2]*dr[1],
                               ang[2]*dr[0]-ang[0]*dr[2],
                               ang[0]*dr[1]-ang[1]*dr[0]};
        const double vrel[3]={v[0]-lin[0]-cross[0],
                              v[1]-lin[1]-cross[1],
                              v[2]-lin[2]-cross[2]};
        const double locx=dot3(dr,ax), locy=dot3(dr,ay);
        const double localvx=dot3(vrel,ax), localvy=dot3(vrel,ay);
        const double gap=dot3(dr,az)-radius_;
        const double lateral=std::max(std::abs(locx),std::abs(locy));
        if(bouncing_) {
          const double rvz=v[2]-lin[2];
          const double tau=clip((rvz+std::sqrt(std::max(0.0,rvz*rvz+
                       2.0*9.81*std::max(gap,0.0))))/9.81,0.0,0.8);
          const double landing=sq((p[0]+tau*v[0]-tip[0]-tau*lin[0])/usable_)
                              +sq((p[1]+tau*v[1]-tip[1]-tau*lin[1])/usable_);
          landing_sum+=landing;
          last_landing=landing;
          const double ball_vin=v[2]-9.81*tau;
          const double desired_tool_v=(desired_vz+e*ball_vin)/(1.0+e);
          const double predicted=(1.0+e)*lin[2]-e*ball_vin;
          const bool approaching=(rvz<0.2 && gap>-0.02);
          const double imminence=0.25+0.75*std::exp(-tau/0.12);
          if(approaching) {
            preimpact_sum+=imminence*sq((lin[2]-desired_tool_v)/std::max(desired_vz,0.5));
            predicted_sum+=imminence*sq((predicted-desired_vz)/desired_vz);
          }
          const bool hit=have_vz && (v[2]-prev_vz>0.12) && (v[2]>0.1) &&
               (gap>-0.025) && (gap<0.04) && (lateral<usable_+0.012);
          if(hit && !prev_hit) {
            ++n_hit;
            impact_sum+=sq((v[2]-desired_vz)/desired_vz);
          }
          prev_hit=hit;
          prev_vz=v[2];
          have_vz=true;
          const bool ascending=v[2]>0.12 && gap>0.01;
          if(ascending) {
            ++n_ascent;
            const double apex=p[2]-tip[2]-radius_+sq(std::max(v[2],0.0))/(2.0*9.81);
            apex_sum+=sq((apex-w[10])/w[10]);
          }
          const bool missed=gap<-0.055 || (lateral>usable_+0.055 && gap<0.05);
          lost_sum+=missed?1.0:0.0;
          any_lost|=missed;
          tilt_sum+=sq((1.0-az[2])/0.03);
          if(t==0 && sub==0) { first_tool_x=tip[0]; first_tool_y=tip[1]; }
          work_sum+=(sq((tip[0]-first_tool_x)/0.18)+sq((tip[1]-first_tool_y)/0.18));
        } else if(sub==substeps-1) {
          const double position=sq(locx/usable_)+sq(locy/usable_);
          const double speed=sq(localvx/0.3)+sq(localvy/0.3);
          position_sum+=position;
          velocity_sum+=speed;
          last_position=position;
          last_velocity=speed;
          tilt_sum+=sq((1.0-az[2])/0.02);
          gap_sum+=sq(std::max(gap-0.015,0.0)/0.03);
          const bool lost=(lateral>usable_) || (gap<-0.035) || (gap>0.085);
          lost_sum+=lost?1.0:0.0;
          any_lost|=lost;
        }
      }
      xyout[t*2]=d->qpos[qadr_];
      xyout[t*2+1]=d->qpos[qadr_+1];
      tdone=t+1;
    }
    if(diverged) {
      const double lastx=d->qpos[qadr_], lasty=d->qpos[qadr_+1];
      for(int t=tdone;t<h;++t) { xyout[2*t]=lastx; xyout[2*t+1]=lasty; }
      job.failed[i]=true;
      job.costs[i]=std::numeric_limits<double>::infinity();
      job.progress[i]=0.;
      return;
    }
    const double H=static_cast<double>(h), F=static_cast<double>(frames);
    double cost;
    if(bouncing_) {
      const double velocity=n_hit>0 ? impact_sum/n_hit : predicted_sum/F;
      const double apex=n_ascent>0 ? apex_sum/n_ascent : 0.0;
      cost=w[0]*(0.6*landing_sum/F+0.4*last_landing)
            +w[2]*preimpact_sum/F+w[1]*velocity+w[3]*apex
            +w[4]*tilt_sum/F+w[5]*work_sum/F
            +w[6]*du_sum/H+w[7]*proposal_sum/H
            +w[8]*(lost_sum/F+(any_lost?0.5:0.0))
            -w[9]*std::min(n_hit,3);
    } else {
      cost=w[0]*position_sum/H +w[1]*velocity_sum/H+w[2]*tilt_sum/H
          +w[5]*gap_sum/H+w[3]*du_sum/H+w[4]*proposal_sum/H
          +w[6]*(last_position+last_velocity)
          +w[7]*(lost_sum/H+(any_lost?0.5:0.0));
    }
    const bool bad=!std::isfinite(cost);
    job.failed[i]=bad;
    job.costs[i]=bad?std::numeric_limits<double>::infinity():cost;
    job.progress[i]=0.;
  }

  mjModel* model_=nullptr;
  std::vector<mjData*> data_;
  int nthread_=1, chunk_=1, state_size_=0;
  bool bouncing_=false;
  int qadr_=0, dadr_=0, sensor_adr_[6]{};
  double motion_scale_=1.0, radius_=0.018, usable_=0.1;
  std::vector<double> vmax_, low_, high_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_, done_;
  std::atomic<int> next_{0};
  const BallJob* job_=nullptr;
  int finished_=0;
  std::uint64_t generation_=0;
  bool stop_=false;
};
}  // namespace

void bind_ball_fused(py::module_& m) {
  py::class_<BallFusedEvaluator>(m,"BallFusedEvaluator")
      .def(py::init<const std::string&,int,int,const std::string&,CArray,double>(),
           py::arg("model_path"),py::arg("nthread"),py::arg("chunk_size"),
           py::arg("task"),py::arg("joint_velocity_limits"),
           py::arg("motion_limit_scale"))
      .def("evaluate",&BallFusedEvaluator::Evaluate,
           py::arg("initial_state"),py::arg("initial_ctrl"),
           py::arg("controls"),py::arg("nominal_controls"),
           py::arg("ctrl_scale"),py::arg("weights"),py::arg("control_substeps"))
      .def_property_readonly("nthread",&BallFusedEvaluator::nthread)
      .def_property_readonly("nstate",&BallFusedEvaluator::nstate)
      .def_property_readonly("chunk_size",&BallFusedEvaluator::chunk_size)
      .def_property_readonly("task_abi",&BallFusedEvaluator::task_abi);
}
