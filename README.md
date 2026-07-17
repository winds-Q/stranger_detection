# 陌生人检测报警系统

利用笔记本摄像头实时检测陌生人，自动发送邮件报警并保存截图。

## 功能

- 📷 实时摄像头采集 + 人脸检测
- 🧑‍🤝‍🧑 熟人注册与识别（欧氏距离比对）
- 📧 陌生人报警邮件（附截图）
- 👥 支持同时向多个目标邮箱发送报警
- 🖥️ Web UI 管理界面（暗色主题）
- ⏱️ 报警冷却机制，防止重复发送
- 🧭 按陌生人临时身份分别冷却，不同陌生人互不影响
- 📂 陌生人截图自动保存与清理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> Windows 下 dlib 安装失败？参考 [启动指南.md](../启动指南.md) 使用预编译 wheel。

### 2. 配置

编辑 `config.yaml`，填写 SMTP 邮箱信息：

```yaml
alert:
  sender_email: "your_email@gmail.com"
  # 授权码仅通过环境变量 STRANGER_DETECTION_SMTP_PASSWORD 提供
  receiver_emails:
    - "first@example.com"
    - "second@example.com"
```

### 3. 注册熟人

将照片放入 `known_faces/` 目录（每人 5~10 张，jpg/png 格式）。

### 4. 启动

**命令行模式：**

```bash
cd src
python main.py
```

**Web UI 模式（推荐）：**

```bash
cd src/web
python app.py
```

浏览器访问 http://localhost:5050

## 项目结构

```
stranger_detection/
├── config.yaml           # 配置文件
├── requirements.txt      # 依赖清单
├── known_faces/          # 熟人照片
├── snapshots/            # 陌生人截图
├── logs/                 # 运行日志
└── src/
    ├── main.py           # 命令行入口
    ├── camera.py         # 摄像头采集
    ├── detector.py       # 人脸检测
    ├── recognizer.py     # 人脸识别
    ├── alerter.py        # 邮件报警
    ├── config_loader.py  # 配置加载
    ├── logger.py         # 日志模块
    └── web/              # Web UI
        ├── app.py
        ├── templates/index.html
        └── static/
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 检测状态 |
| `/api/detect/start` | POST | 启动检测 |
| `/api/detect/stop` | POST | 停止检测 |
| `/api/faces` | GET | 熟人列表 |
| `/api/faces/upload` | POST | 上传照片 |
| `/api/faces/<name>` | DELETE | 删除照片 |
| `/api/logs/stream` | GET | SSE 日志流 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `STRANGER_DETECTION_CONFIG` | 自定义配置文件路径 |
| `STRANGER_DETECTION_SMTP_PASSWORD` | SMTP 密码（优先于 config.yaml） |
| `STRANGER_DETECTION_WEB_HOST` | Web 监听地址，默认 `127.0.0.1` |
| `STRANGER_DETECTION_WEB_PORT` | Web 监听端口，默认 `5050` |

## 测试

```bash
python -m unittest discover -s tests -v
```

Web 服务默认仅允许本机访问。如确实需要局域网访问，可将
`STRANGER_DETECTION_WEB_HOST` 设置为 `0.0.0.0`，并自行配置防火墙和访问控制。

## License

MIT
