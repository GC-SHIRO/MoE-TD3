# 单帧 marker 检测结果。没有 marker 时也必须发布（visible=false），
# 用来区分“看不到 marker”和“没有收到视觉帧”。
std_msgs/Header header
bool visible
# 无人机相对 marker 的位置，米，轴向与世界 ENU 对齐（偏航固定为 0 时即机体轴）
geometry_msgs/Vector3 position
float32 confidence
