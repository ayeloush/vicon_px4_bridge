#include "geometry_msgs/msg/transform_stamped.hpp"
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <px4_msgs/msg/timesync_status.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/utils.h>

using namespace std::chrono_literals;
using std::placeholders::_1;

class ViconPX4Bridge : public rclcpp::Node {
public:
  ViconPX4Bridge() : Node("vicon_px4_bridge") {
    this->declare_parameter<std::string>("px4_name", "radd");
    this->declare_parameter<std::string>("vicon_name", "radd");
    std::string vicon_name;
    this->get_parameter("px4_name", px4_name);
    this->get_parameter("vicon_name", vicon_name);

    // Timing. timestamp_sample is the Vicon frame time (the vicon_bridge header stamp: grab time minus
    // the SDK's reported latency), so PX4's EKF fuses each pose at the time it was measured even when
    // this bridge falls behind; EKF2_EV_DELAY then only covers what that stamp leaves out (~0-5 ms).
    // Needs the vicon_bridge and this node on the same clock (same machine, or chrony-synced).
    use_vicon_stamp_ = this->declare_parameter<bool>("use_vicon_stamp", true);
    // Drop poses older than this: the EKF's delayed horizon is EKF2_DELAY_MAX (default 200 ms).
    max_sample_age_ms_ = this->declare_parameter<double>("max_sample_age_ms", 150.0);
    // A header stamp further than this in the future means the clocks disagree: stamp on receipt.
    max_future_ms_ = this->declare_parameter<double>("max_future_ms", 5.0);
    // Forward every n-th pose (1 = all; PX4's EKF accepts vision samples down to ~7 ms apart).
    downsample_ = std::max<int64_t>(1, this->declare_parameter<int64_t>("downsample", 1));
    std::string vicon_sub_name = "vicon/" + vicon_name + "/" + vicon_name;

    std::string timesync_sub_name = px4_name + "/fmu/out/timesync";
    //std::string px4_pub_name = px4_name + "/fmu/in/vehicle_visual_odometry";
    std::string px4_pub_name = "/fmu/in/vehicle_visual_odometry";

    RCLCPP_INFO(this->get_logger(), "vicon_sub_name: %s \n",
                vicon_sub_name.c_str());
    RCLCPP_INFO(this->get_logger(), "timesync_sub_name: %s \n",
                timesync_sub_name.c_str());
    RCLCPP_INFO(this->get_logger(), "px4_pub_name: %s \n",
                px4_pub_name.c_str());
    RCLCPP_INFO(this->get_logger(), "Using Vicon -> Visual Odometry callback!");

    // Sensor-data QoS with a short queue: with per-pose stamps a late pose is still useful, and a
    // depth-1 queue silently drops poses whenever a callback runs late.
    position_sub_ =
        this->create_subscription<geometry_msgs::msg::TransformStamped>(
            vicon_sub_name, rclcpp::SensorDataQoS().keep_last(10),
            std::bind(&ViconPX4Bridge::position_callback, this, _1));

    odometry_pub_ = this->create_publisher<px4_msgs::msg::VehicleOdometry>(
        px4_pub_name, 10);

    // timesync_sub_ = this->create_subscription<px4_msgs::msg::TimesyncStatus>(
    //     timesync_sub_name, 10,
    //     [this](const px4_msgs::msg::TimesyncStatus::UniquePtr msg) {
    //       this->px4_timestamp_.store(msg->timestamp);
    //       this->px4_server_timestamp_.store(
    //           this->get_clock()->now().nanoseconds());
    //     });
  }

private:
  std::string px4_name;

  rclcpp::Publisher<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_pub_;

  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr
      position_sub_;
  rclcpp::Subscription<px4_msgs::msg::TimesyncStatus>::SharedPtr timesync_sub_;
  std::atomic<uint64_t> px4_timestamp_;
  std::atomic<uint64_t> px4_server_timestamp_;

  uint64_t counter_ = 0;
  bool use_vicon_stamp_ = true;
  double max_sample_age_ms_ = 150.0;
  double max_future_ms_ = 5.0;
  int64_t downsample_ = 1;

  // Pose age (receipt minus Vicon frame time) statistics, reported every 5 s.
  std::vector<double> ages_ms_;
  int64_t last_report_us_ = 0;
  uint64_t dropped_old_ = 0, clock_mismatch_ = 0;

  void report_ages(int64_t now_us) {
    if (last_report_us_ == 0) {
      last_report_us_ = now_us;
      return;
    }
    if (now_us - last_report_us_ < 5000000) {
      return;
    }
    if (!ages_ms_.empty()) {
      std::sort(ages_ms_.begin(), ages_ms_.end());
      RCLCPP_INFO(this->get_logger(),
                  "vicon pose age at bridge [ms]: median %.1f, p95 %.1f, max "
                  "%.1f (n=%zu); dropped as too old: %lu, clock mismatch: %lu",
                  ages_ms_[ages_ms_.size() / 2],
                  ages_ms_[(ages_ms_.size() * 95) / 100], ages_ms_.back(),
                  ages_ms_.size(), dropped_old_, clock_mismatch_);
    }
    ages_ms_.clear();
    last_report_us_ = now_us;
  }

  uint64_t get_current_timestamp() {
    auto delta =
        (this->get_clock()->now().nanoseconds() - this->px4_server_timestamp_) /
        1e6;
    return px4_timestamp_.load() + delta;
  }

  void
  position_callback(const geometry_msgs::msg::TransformStamped &vicon_msg) {

    auto message = px4_msgs::msg::VehicleOdometry();
    counter_++;
    if (counter_ % downsample_ != 0) {
      return;
    }

    // timestamps: PX4's uXRCE-DDS client converts both from this machine's ROS
    // time to its own clock (timesync offset).
    const int64_t now_us = this->get_clock()->now().nanoseconds() / 1000;
    message.timestamp = now_us;
    message.timestamp_sample = now_us;
    if (use_vicon_stamp_) {
      const int64_t stamp_us =
          static_cast<int64_t>(vicon_msg.header.stamp.sec) * 1000000 +
          vicon_msg.header.stamp.nanosec / 1000;
      const double age_ms = (now_us - stamp_us) / 1000.0;
      report_ages(now_us);
      if (age_ms > max_sample_age_ms_) {
        dropped_old_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "dropping vicon pose %.0f ms old (max_sample_age_ms %.0f)",
                             age_ms, max_sample_age_ms_);
        return;
      }
      if (age_ms < -max_future_ms_) {
        clock_mismatch_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "vicon header stamp is %.0f ms in the future: the vicon_bridge "
                             "clock disagrees with this machine's; stamping on receipt",
                             -age_ms);
      } else {
        message.timestamp_sample = std::min(stamp_us, now_us);
        ages_ms_.push_back(age_ms);
      }
    }

    // pose frame
    message.pose_frame = px4_msgs::msg::VehicleOdometry::POSE_FRAME_NED;

    // position
    message.position[0] = vicon_msg.transform.translation.y;
    message.position[1] = vicon_msg.transform.translation.x;
    message.position[2] = -vicon_msg.transform.translation.z;

    // quaternion
    auto q_vicon = vicon_msg.transform.rotation;
    tf2::Quaternion tf2_vicon(q_vicon.x, q_vicon.y, q_vicon.z, q_vicon.w);
    // auto yaw = tf2::getYaw(tf2_vicon);
    // auto yaw_ned = -yaw + 1.57;
    // Eigen::Quaternion<double> out(cos(yaw_ned / 2.0), 0, 0, sin(yaw_ned / 2.0));

    // message.q[0] = out.w();
    // message.q[1] = out.x();
    // message.q[2] = out.y();
    // message.q[3] = out.z();

  double roll_vicon, pitch_vicon, yaw_vicon;
  tf2::Matrix3x3(tf2_vicon).getRPY(roll_vicon, pitch_vicon, yaw_vicon);

  double roll_ned  = roll_vicon;
  double pitch_ned = -pitch_vicon;
  double yaw_ned   = -yaw_vicon + 1.5708;

  tf2::Quaternion q_ned;
  q_ned.setRPY(roll_ned, pitch_ned, yaw_ned);

  message.q[0] = q_ned.getW();
  message.q[1] = q_ned.getX();
  message.q[2] = q_ned.getY();
  message.q[3] = q_ned.getZ();

    // velocity frame
    message.velocity_frame =
        px4_msgs::msg::VehicleOdometry::VELOCITY_FRAME_UNKNOWN;

    for (size_t i = 0; i < 3; i++) {

      // velocity
      message.velocity[i] = NAN;

      // angular velocity
      message.angular_velocity[i] = NAN;
    }

    // position covariance
    for (size_t i = 0; i < 3; i++) {
      message.position_variance[i] = std::pow(0.01, 2);
      message.orientation_variance[i] = std::pow(0.0523, 2); // approx 3 degrees
      message.velocity_variance[i] = NAN;
    }

    message.reset_counter = 0;

    odometry_pub_->publish(message);
  }
};

int main(int argc, char *argv[]) {
  std::cout << "Starting vicon px4 bridge node..." << std::endl;
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ViconPX4Bridge>());

  rclcpp::shutdown();
  return 0;
}
