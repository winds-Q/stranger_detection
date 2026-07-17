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
- 📊 Web 健康状态面板，展示摄像头、检测 FPS、告警数量和邮件状态
- 🗃️ 使用 SQLite 持久化告警事件，重启程序后仍可查询
- 🕘 Web 告警历史支持查看截图、标记已处理和删除
- 🧹 按保留天数自动清理截图、日志和历史事件
- ✅ 上传熟人照片时自动检查人脸数量、尺寸、清晰度、亮度和重复人脸
- 🔐 SMTP 授权码只从环境变量读取，并支持在 Web 端发送测试邮件

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> Windows 下 dlib 安装失败？参考 [启动指南.md](启动指南.md) 使用预编译 wheel。

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

也可以在 Web 页面填写熟人姓名后上传。系统会为同一姓名的多张样本生成唯一文件名，但识别和日志统一显示填写的姓名。上传时系统会拒绝无人脸、多张人脸、人脸过小、过暗、过亮、模糊或与已有样本重复的照片。

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
├── data/alerts.db        # SQLite 告警事件数据库（运行后自动创建）
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
| `/api/cameras` | GET | 当前摄像头配置 |
| `/api/cameras/scan` | POST | 扫描可用摄像头 |
| `/api/cameras/select` | POST | 选择摄像头 |
| `/api/events` | GET | 查询告警历史 |
| `/api/events/<id>/handled` | POST | 标记事件已处理 |
| `/api/events/<id>` | DELETE | 删除事件及其截图 |
| `/api/events/<id>/snapshot` | GET | 查看事件截图 |
| `/api/detect/start` | POST | 启动检测 |
| `/api/detect/stop` | POST | 停止检测 |
| `/api/faces` | GET | 熟人列表 |
| `/api/faces/upload` | POST | 上传照片 |
| `/api/faces/<name>` | DELETE | 删除照片 |
| `/api/logs/stream` | GET | SSE 日志流 |
| `/api/config` | GET/POST | 查看或保存非敏感配置 |
| `/api/config/test-email` | POST | 发送 SMTP 测试邮件 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `STRANGER_DETECTION_CONFIG` | 自定义配置文件路径 |
| `STRANGER_DETECTION_SMTP_PASSWORD` | SMTP 密码/授权码；程序只从该变量读取，不写入配置文件或 Web 接口 |
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
