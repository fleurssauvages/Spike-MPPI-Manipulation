// Separate, reaching-only native fused velocity rollout backend.
// Command-shaper dynamics: low_level joint_velocity_controller.cpp, 1 kHz.
// Kinematic integration + mj_forward (not a physical contact dynamics model).
#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace py = pybind11;
using A = py::array_t<double, py::array::c_style | py::array::forcecast>;
class FusedVelocityReaching {
 mjModel* model_ = nullptr;
 int tip_geom_ = -1;
 int qadr_[7], dadr_[7];
 public:
 explicit FusedVelocityReaching(const std::string& path, const std::string& tip_name,
                                  py::array_t<int,py::array::c_style | py::array::forcecast> qa,
                                  py::array_t<int,py::array::c_style | py::array::forcecast> da) {
   if (qa.size()!=7 || da.size()!=7) throw std::runtime_error("expected seven arm addresses");
   model_=mj_loadXML(path.c_str(),nullptr,nullptr,0);
   if(!model_) throw std::runtime_error("cannot load model XML; verify included meshes/paths: "+path);
   tip_geom_=mj_name2id(model_,mjOBJ_GEOM,tip_name.c_str());
   if(tip_geom_<0) throw std::runtime_error("missing tip geom: "+tip_name);
   for(int j=0;j<7;j++) {qadr_[j]=qa.data()[j]; dadr_[j]=da.data()[j];
     if(qadr_[j]<0 || qadr_[j]>=model_->nq || dadr_[j]<0 || dadr_[j]>=model_->nv)
       throw std::runtime_error("invalid arm index");}
 }
 ~FusedVelocityReaching(){if(model_) mj_deleteModel(model_);}
 FusedVelocityReaching(const FusedVelocityReaching&)=delete;
 FusedVelocityReaching& operator=(const FusedVelocityReaching&)=delete;
 py::tuple evaluate(A controls, A qpos, A qvel, A target, A nominal,
                    double kp=2., double kd=.2, double vmax=.4, double outer_amax=.6,
                    double low_amax=2., double jerk=100., double tau=.03,
                    double period=.01, double dt=.001,
                    double radius=.025, double region_weight=1.,
                    double terminal_weight=1., double velocity_weight=0.,
                    double velocity_sigma=.1, double velocity_ref=.1,
                    double settle_weight=0., int settle_steps=5,
                    double accel_weight=0., double control_weight=0.) const {
   if(controls.ndim()!=3 || controls.shape(2)!=7 || qpos.size()!=model_->nq || qvel.size()!=model_->nv
      || target.size()!=3 || nominal.ndim()!=2 || nominal.shape(0)!=controls.shape(1)
      || nominal.shape(1)!=7) throw std::runtime_error("invalid input array shape");
   const int n=(int)controls.shape(0), h=(int)controls.shape(1);
   const int ticks=(int)std::llround(period/dt);
   if(n<1||h<1||ticks<1||std::abs(ticks*dt-period)>1e-10 || !std::isfinite(kp)|| !std::isfinite(kd)
     || !(vmax>0) || !(outer_amax>0)|| !(low_amax>0)||!(jerk>0)||!(tau>0)||!(radius>0)||!(velocity_ref>0))
     throw std::runtime_error("invalid control/step parameters");
   py::array_t<double> xy({n,h,2}), costs(n), progress(n);
   py::array_t<bool> failed(n);
   const auto u=controls.unchecked<3>(); auto out=xy.mutable_unchecked<3>();
   auto cc=costs.mutable_unchecked<1>();auto pp=progress.mutable_unchecked<1>();auto ff=failed.mutable_unchecked<1>();
   const double* targ=target.data();const double* nom=nominal.data();
   {
    py::gil_scoped_release unlock;
    // Each worker has its own mjData; avoid per-timestep allocations and calls into Python.
    #pragma omp parallel
    {
     mjData* d=mj_makeData(model_);
     #pragma omp for schedule(static)
     for(int i=0;i<n;i++) {
      mj_resetData(model_,d);
      std::memcpy(d->qpos,qpos.data(),model_->nq*sizeof(double));
      std::memcpy(d->qvel,qvel.data(),model_->nv*sizeof(double));
      double v[7]={}, a[7]={}, prev[7]={};
      for(int j=0;j<7;j++) v[j]=d->qvel[dadr_[j]];
      double integral=0.,settle=0., lastdist=0.,lastv2=0., dev=0.,accelcost=0.;
      double prior_tip[3]={};bool has_prior=false;
      bool bad=false;
      for(int t=0;t<h;t++) {
        double req[7];
        for(int j=0;j<7;j++) {
          const double val=kp*(u(i,t,j)-d->qpos[qadr_[j]])-kd*d->qvel[dadr_[j]];
          req[j]=std::clamp(val,-vmax,vmax);
          req[j]=std::clamp(req[j],prev[j]-outer_amax*period,prev[j]+outer_amax*period);
          prev[j]=req[j];
          const double dc=u(i,t,j)-nom[t*7+j];dev+=dc*dc/7.;
          if(t>=2){const double second=u(i,t,j)-2*u(i,t-1,j)+u(i,t-2,j);accelcost+=second*second/7.;}
        }
        for(int sub=0;sub<ticks;sub++) {
          double desired[7],delta[7];double scalea=1.,scalej=1.;
          for(int j=0;j<7;j++){desired[j]=(req[j]-v[j])/tau;scalea=std::max(scalea,std::abs(desired[j])/low_amax);}
          for(int j=0;j<7;j++){desired[j]/=scalea;delta[j]=desired[j]-a[j];scalej=std::max(scalej,std::abs(delta[j])/(jerk*dt));}
          for(int j=0;j<7;j++) {a[j]+=delta[j]/scalej;v[j]+=a[j]*dt;d->qpos[qadr_[j]]+=v[j]*dt;}
        }
        for(int j=0;j<7;j++) d->qvel[dadr_[j]]=v[j];
        d->time+=period;
        mj_forward(model_,d);
        const double* xyz=d->geom_xpos+3*tip_geom_;
        out(i,t,0)=xyz[0];out(i,t,1)=xyz[1];
        double dist2=0., vel2=0.;
        for(int j=0;j<3;j++) {
          const double diff=xyz[j]-targ[j];dist2+=diff*diff;
          if(has_prior) {const double dv=(xyz[j]-prior_tip[j])/period;vel2+=dv*dv;}
          prior_tip[j]=xyz[j];
        }
        has_prior=true;
        const double distance=std::sqrt(dist2);
        const double outside=std::max(0.,distance/radius-1.);
        integral+=region_weight*outside*outside;
        integral+=velocity_weight*std::exp(-dist2/(2*velocity_sigma*velocity_sigma))*vel2/(velocity_ref*velocity_ref);
        if(t>=h-std::max(1,settle_steps))settle+=dist2/(radius*radius)+vel2/(velocity_ref*velocity_ref);
        lastdist=distance;lastv2=vel2;
        for(int j=0;j<7;j++) if(!std::isfinite(d->qpos[qadr_[j]]) || !std::isfinite(v[j]))bad=true;
        if(bad) break;
      }
      cc(i)=bad?INFINITY:(integral/h+terminal_weight*lastdist/std::max(radius,.02)
            + settle_weight*settle/std::max(1,std::min(h,settle_steps))
            + control_weight*dev/h+accel_weight*accelcost/h);
      pp(i)=0.;ff(i)=bad;
     }
     mj_deleteData(d);
    }
   }
   return py::make_tuple(xy,costs,progress,failed);
 }
};
PYBIND11_MODULE(_fused_velocity,m){
  m.doc()="Separate C++ 1-kHz-shaper reaching rollout; kinematic proxy, not full FR3 mechanics";
  py::class_<FusedVelocityReaching>(m,"FusedVelocityReaching")
    .def(py::init<const std::string&,const std::string&,
       py::array_t<int,py::array::c_style|py::array::forcecast>,
       py::array_t<int,py::array::c_style|py::array::forcecast>>())
    .def("evaluate",&FusedVelocityReaching::evaluate,
       py::arg("controls"),py::arg("qpos"),py::arg("qvel"),py::arg("target"),py::arg("nominal"),
       py::arg("kp")=2.,py::arg("kd")=.2,py::arg("vmax")=.4,py::arg("outer_amax")=.6,
       py::arg("low_amax")=2.,py::arg("jerk")=100.,py::arg("tau")=.03,
       py::arg("period")=.01,py::arg("dt")=.001,py::arg("radius")=.025,
       py::arg("region_weight")=1.,py::arg("terminal_weight")=1.,py::arg("velocity_weight")=0.,
       py::arg("velocity_sigma")=.1,py::arg("velocity_ref")=.1,py::arg("settle_weight")=0.,
       py::arg("settle_steps")=5,py::arg("accel_weight")=0.,py::arg("control_weight")=0.);
}
