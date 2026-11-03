# Build Script 使用说明

本项目提供了一个 `build.sh` 脚本用于简化编译流程。

---
## 📌 基本用法
```bash
# 赋予脚本执行权限
chmod +x build.sh

# 使用默认参数编译
./build.sh


⚙️ 参数选项
参数	         缩写	   说明	                       默认值
--rocm-path	    -p	      设置 ROCm 安装路径	       $ROCM_PATH 环境变量
--build-tests	-t	      启用/禁用测试构建 (ON/OFF)	ON
--full-stress	-f	      启用/禁用全压力测试 (ON/OFF)	 OFF
--help	        -h	      显示帮助信息	                  -
```

🖥️ 使用示例
```bash
# 指定 ROCm 路径
./build.sh --rocm-path /opt/rocm

# 禁用测试构建
./build.sh --build-tests OFF

# 启用测试和全压力测试
./build.sh --build-tests ON --full-stress ON

# 使用短参数形式
./build.sh -p /opt/rocm -t ON -f ON

# 组合使用参数
./build.sh -p /opt/rocm-5.7.0 -t OFF
```

## 🔧 环境要求

1. 已正确source dtk/env.sh
2. CMake ≥ 3.10