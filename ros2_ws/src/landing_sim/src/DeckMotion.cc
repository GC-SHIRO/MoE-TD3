// 甲板运动插件：按仿真时间设置船体模型位姿（运动学），公式与 core/ship_motion.py 完全一致。
// 两边任何一处修改都必须同步，并更新 tests/test_ship_motion.py 中的格式测试。
//
// 命令（gz.msgs.StringMsg，空格分隔），由 core.ship_motion.encode_plugin_command 生成：
//   start_time x0 y0 z0 heading n_seg [t v]*n_seg n_wave [axis amplitude period phase]*n_wave
//   axis：0=heave（米），1=roll（弧度），2=pitch（弧度）
// start_time 之前保持场景 t=0 的位姿，复位期间甲板不会跳变。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

namespace landing
{
struct Wave
{
  int axis{0};
  double amplitude{0.0};
  double period{1.0};
  double phase{0.0};
};

struct Command
{
  double start{0.0};
  double x0{0.0};
  double y0{0.0};
  double z0{0.0};
  double heading{0.0};
  std::vector<std::pair<double, double>> schedule;
  std::vector<Wave> waves;
};

class DeckMotion : public gz::sim::System,
                   public gz::sim::ISystemConfigure,
                   public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &,
                 gz::sim::EventManager &) override
  {
    this->model = gz::sim::Model(entity);
    const std::string topic =
      sdf->Get<std::string>("command_topic", "/landing/deck_command").first;
    this->node.Subscribe(topic, &DeckMotion::OnCommand, this);
  }

  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override
  {
    if (info.paused)
      return;
    std::lock_guard<std::mutex> lock(this->mutex);
    if (!this->hasCommand)
      return;
    const double now = std::chrono::duration<double>(info.simTime).count();
    const double t = std::max(0.0, now - this->command.start);
    this->model.SetWorldPoseCmd(ecm, this->Pose(t));
  }

private:
  void OnCommand(const gz::msgs::StringMsg &msg)
  {
    std::istringstream stream(msg.data());
    Command parsed;
    std::size_t segments = 0;
    std::size_t waves = 0;
    stream >> parsed.start >> parsed.x0 >> parsed.y0 >> parsed.z0 >> parsed.heading >> segments;
    for (std::size_t i = 0; i < segments; ++i)
    {
      double t = 0.0, v = 0.0;
      stream >> t >> v;
      parsed.schedule.emplace_back(t, v);
    }
    stream >> waves;
    for (std::size_t i = 0; i < waves; ++i)
    {
      Wave wave;
      stream >> wave.axis >> wave.amplitude >> wave.period >> wave.phase;
      parsed.waves.push_back(wave);
    }
    if (stream.fail())
    {
      gzerr << "DeckMotion 命令解析失败: " << msg.data() << std::endl;
      return;
    }
    std::lock_guard<std::mutex> lock(this->mutex);
    this->command = parsed;
    this->hasCommand = true;
  }

  // 分段常速的解析积分，对应 DeckMotion._distance。
  double Distance(double t) const
  {
    double distance = 0.0;
    const auto &schedule = this->command.schedule;
    for (std::size_t i = 0; i < schedule.size(); ++i)
    {
      const double start = schedule[i].first;
      if (t <= start)
        break;
      const double end = (i + 1 < schedule.size()) ? schedule[i + 1].first
                                                   : std::numeric_limits<double>::infinity();
      distance += schedule[i].second * (std::min(t, end) - start);
    }
    return distance;
  }

  double WaveValue(int axis, double t) const
  {
    double value = 0.0;
    for (const auto &wave : this->command.waves)
    {
      if (wave.axis != axis)
        continue;
      const double omega = 2.0 * M_PI / wave.period;
      value += wave.amplitude * std::sin(omega * t + wave.phase);
    }
    return value;
  }

  gz::math::Pose3d Pose(double t) const
  {
    const double distance = this->Distance(t);
    const double x = this->command.x0 + std::cos(this->command.heading) * distance;
    const double y = this->command.y0 + std::sin(this->command.heading) * distance;
    const double z = this->command.z0 + this->WaveValue(0, t);
    // 船体偏航固定为 0；gz 的 (roll, pitch, yaw) 与 core.frames.rotation_from_rpy 同为 Z-Y-X 顺序。
    return gz::math::Pose3d(x, y, z, this->WaveValue(1, t), this->WaveValue(2, t), 0.0);
  }

  gz::sim::Model model{gz::sim::kNullEntity};
  gz::transport::Node node;
  std::mutex mutex;
  Command command;
  bool hasCommand{false};
};
}  // namespace landing

GZ_ADD_PLUGIN(landing::DeckMotion,
              gz::sim::System,
              landing::DeckMotion::ISystemConfigure,
              landing::DeckMotion::ISystemPreUpdate)
