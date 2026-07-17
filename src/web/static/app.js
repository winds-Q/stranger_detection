const logViewer = document.getElementById("logViewer");
const btnStart  = document.getElementById("btnStart");
const btnStop   = document.getElementById("btnStop");
const statusDot = document.getElementById("statusDot");
const statusTxt = document.getElementById("statusText");
const cameraSelect = document.getElementById("cameraSelect");
const btnScanCamera = document.getElementById("btnScanCamera");
const btnSaveCamera = document.getElementById("btnSaveCamera");

let logLines = [];
let logEventSource = null;

function updateButtons(running) {
    btnStart.disabled = running;
    btnStop.disabled  = !running;
    statusDot.className = running ? "dot on" : "dot off";
    statusTxt.textContent = running ? "运行中" : "未运行";
    btnScanCamera.disabled = running;
    btnSaveCamera.disabled = running || !cameraSelect.value;
}

function updateHealth(data) {
    const cameraLabels = {
        connected: "已连接",
        reconnecting: "重连中",
        disconnected: "未连接",
    };
    document.getElementById("healthCamera").textContent = cameraLabels[data.camera_state] || data.camera_state;
    document.getElementById("healthFps").textContent = Number(data.detection_fps || 0).toFixed(1);
    document.getElementById("healthFaces").textContent = data.known_faces ?? 0;
    document.getElementById("healthAlerts").textContent = data.alerts_today ?? 0;
    document.getElementById("healthSmtp").textContent = data.smtp_configured ? "已配置" : "未配置";
    document.getElementById("healthQueue").textContent = data.alert_queue_pending ?? 0;
}

async function loadCameraSelection() {
    try {
        const res = await fetch("/api/cameras");
        const data = await res.json();
        cameraSelect.innerHTML = `<option value="${data.selected_device_id}">摄像头 ${data.selected_device_id}（当前配置）</option>`;
        cameraSelect.value = String(data.selected_device_id);
        btnSaveCamera.disabled = true;
    } catch (e) {
        document.getElementById("cameraStatus").textContent = "无法读取摄像头配置";
    }
}

async function scanCameras() {
    const statusEl = document.getElementById("cameraStatus");
    btnScanCamera.disabled = true;
    btnSaveCamera.disabled = true;
    statusEl.textContent = "正在扫描摄像头，请稍候...";
    try {
        const res = await fetch("/api/cameras/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ max_devices: 6 }),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
            statusEl.textContent = data.message || "扫描失败";
            return;
        }
        if (!data.devices.length) {
            cameraSelect.innerHTML = '<option value="">未发现可用摄像头</option>';
            statusEl.textContent = "未发现可用摄像头，请检查权限或设备连接";
            return;
        }
        cameraSelect.innerHTML = data.devices.map(device =>
            `<option value="${device.device_id}">${escapeHtml(device.label)}</option>`
        ).join("");
        btnSaveCamera.disabled = false;
        statusEl.textContent = `发现 ${data.devices.length} 个可用设备`;
    } catch (e) {
        statusEl.textContent = "扫描请求失败，请检查 Web 服务";
    } finally {
        btnScanCamera.disabled = false;
    }
}

async function saveCameraSelection() {
    const statusEl = document.getElementById("cameraStatus");
    if (!cameraSelect.value) return;
    btnSaveCamera.disabled = true;
    statusEl.textContent = "正在保存...";
    try {
        const res = await fetch("/api/cameras/select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device_id: Number(cameraSelect.value) }),
        });
        const data = await res.json();
        statusEl.textContent = data.message || (res.ok ? "摄像头已保存" : "保存失败");
    } catch (e) {
        statusEl.textContent = "保存失败，请检查 Web 服务";
        btnSaveCamera.disabled = false;
    }
}

async function loadEvents() {
    const list = document.getElementById("eventList");
    try {
        const res = await fetch("/api/events?limit=30");
        const events = await res.json();
        if (!events.length) {
            list.innerHTML = '<div class="empty-state">暂无告警记录</div>';
            return;
        }
        const statusLabels = { pending: "待发送", sent: "已发送", failed: "发送失败" };
        const statusClasses = { pending: "status-pending", sent: "status-sent", failed: "status-failed" };
        list.innerHTML = events.map(event => {
          const statusClass = statusClasses[event.notification_status] || "status-pending";
          const snapshotUrl = `/api/events/${event.id}/snapshot`;
          return `
          <div class="event-item ${statusClass}">
            ${event.has_snapshot ? `<a class="event-snapshot" href="${snapshotUrl}" target="_blank" rel="noopener" title="点击查看陌生人照片"><img src="${snapshotUrl}" alt="陌生人告警截图" loading="lazy" onerror="handleSnapshotError(this)"><span class="snapshot-zoom">查看原图</span></a>` : `<div class="event-snapshot event-snapshot-empty">暂无照片</div>`}
            <div class="event-top"><span class="event-id" title="${escapeHtml(event.event_key)}">${escapeHtml(event.stranger_id)}</span><span class="event-status ${statusClass}">${statusLabels[event.notification_status] || escapeHtml(event.notification_status)}</span></div>
            <div class="event-meta">出现：${escapeHtml(event.first_seen_at)}<br>离开：${escapeHtml(event.left_at || "仍在画面或未确认")}</div>
            <div class="event-actions">
              ${event.has_snapshot ? `<a class="btn btn-small event-photo-link" href="${snapshotUrl}" target="_blank" rel="noopener">查看照片</a>` : ""}
              <button class="btn btn-small" onclick="markEventHandled(${event.id}, ${!event.handled})">${event.handled ? "取消处理" : "标记处理"}</button>
              <button class="btn btn-red" onclick="deleteEvent(${event.id})">删除</button>
            </div>
          </div>`;
        }).join("");
    } catch (e) {
        list.innerHTML = '<div class="empty-state">告警历史加载失败</div>';
    }
}

function handleSnapshotError(image) {
    const container = image.closest(".event-snapshot");
    if (container) {
        container.classList.add("event-snapshot-error");
        container.removeAttribute("href");
    }
    image.remove();
}

async function markEventHandled(eventId, handled) {
    await fetch(`/api/events/${eventId}/handled`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({handled})});
    loadEvents();
}

async function deleteEvent(eventId) {
    if (!confirm("确定删除此告警记录及关联截图？")) return;
    await fetch(`/api/events/${eventId}?delete_snapshot=1`, {method: "DELETE"});
    loadEvents();
}

cameraSelect.addEventListener("change", () => {
    btnSaveCamera.disabled = !cameraSelect.value || btnStart.disabled;
});

function appendLog(text) {
    let cls = "info";
    if (text.includes("[ERROR]"))     cls = "error";
    else if (text.includes("[WARNING]")) cls = "warn";
    else if (text.includes("[DEBUG]"))   cls = "debug";

    const div = document.createElement("div");
    div.className = "log-line";
    div.innerHTML = `<span class="${cls}">${escapeHtml(text)}</span>`;

    logViewer.appendChild(div);
    logViewer.scrollTop = logViewer.scrollHeight;
}

function escapeHtml(s) {
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function clearLogs() {
    logViewer.innerHTML = "";
}

async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        updateButtons(data.running);
        updateHealth(data);
        if (data.error && !data.running) {
            statusTxt.textContent = "启动失败";
            statusTxt.title = data.error;
        }
    } catch (e) {
        console.error("状态查询失败", e);
    }
}

async function startDetection() {
    try {
        const res = await fetch("/api/detect/start", { method: "POST" });
        const data = await res.json();
        if (data.ok) {
            updateButtons(true);
        } else {
            updateButtons(false);
            alert(data.message);
        }
    } catch (e) {
        alert("请求失败: " + e);
    }
}

async function stopDetection() {
    try {
        const res = await fetch("/api/detect/stop", { method: "POST" });
        const data = await res.json();
        if (data.ok) {
            updateButtons(false);
        } else {
            alert(data.message);
        }
    } catch (e) {
        alert("请求失败: " + e);
    }
}

async function handleUpload(input) {
    const files = input.files;
    if (!files.length) return;

    const statusEl = document.getElementById("uploadStatus");
    const personName = document.getElementById("knownPersonName").value.trim();
    if (!personName) {
        statusEl.textContent = "请先填写熟人姓名";
        input.value = "";
        return;
    }
    statusEl.classList.remove("error");
    statusEl.textContent = "上传中...";

    let succeeded = 0;
    const failureMessages = [];
    const selectedFiles = Array.from(files);
    for (let index = 0; index < selectedFiles.length; index++) {
        const file = selectedFiles[index];
        statusEl.textContent = `正在检查第 ${index + 1}/${selectedFiles.length} 张：${file.name}`;
        const form = new FormData();
        form.append("file", file);
        form.append("person_name", personName);
        try {
            const response = await fetch("/api/faces/upload", { method: "POST", body: form });
            const data = await response.json();
            if (response.ok && data.ok) {
                succeeded++;
            } else {
                failureMessages.push(`${file.name}：${data.message || "上传失败"}`);
            }
        } catch (_) {
            failureMessages.push(`${file.name}：无法连接 Web 服务，请确认服务仍在运行`);
        }
    }
    const failed = selectedFiles.length - succeeded;
    statusEl.textContent = failed
        ? `${succeeded} 个成功，${failed} 个失败\n${failureMessages.join("\n")}`
        : `已上传 ${succeeded} 个文件`;
    statusEl.classList.toggle("error", failed > 0);
    loadFaces();
    input.value = "";
}

async function openFacesFolder() {
    try {
        const res = await fetch("/api/faces/open-folder", { method: "POST" });
        const data = await res.json();
        if (!res.ok || !data.ok) alert(data.message || "无法打开熟人目录");
    } catch (_) {
        alert("无法连接 Web 服务");
    }
}

async function deleteFace(filename) {
    if (!confirm(`确定删除 ${filename}？`)) return;
    try {
        const res = await fetch("/api/faces/" + encodeURIComponent(filename), { method: "DELETE" });
        const data = await res.json();
        if (data.ok) {
            loadFaces();
        } else {
            alert(data.message);
        }
    } catch (e) {
        alert("请求失败: " + e);
    }
}

async function loadFaces() {
    try {
        const res = await fetch("/api/faces");
        const items = await res.json();
        const list = document.getElementById("faceList");
        const count = document.getElementById("faceCount");

        if (!items.length) {
            list.innerHTML = "<div style='color:#6e7681;font-size:12px'>暂无熟人照片</div>";
        } else {
            list.innerHTML = items.map(f => {
                const safeName = escapeHtml(f.name);
                const personName = escapeHtml(f.person_name || f.name);
                return `<div class="face-item">
                    <span class="name" title="样本文件：${safeName}"><strong>${personName}</strong><small>${safeName}</small></span>
                    <button class="delete-btn" title="删除" data-filename="${safeName}">&times;</button>
                </div>`;
            }).join("");
            // 使用事件委托替代内联 onclick
            list.querySelectorAll(".delete-btn").forEach(btn => {
                btn.addEventListener("click", () => deleteFace(btn.dataset.filename));
            });
        }
        count.textContent = items.length;
    } catch (e) {
        console.error("加载熟人列表失败", e);
    }
}

function startLogStream() {
    if (logEventSource && logEventSource.readyState !== EventSource.CLOSED) {
        return;
    }

    logEventSource = new EventSource("/api/logs/stream");
    logEventSource.onmessage = function (e) {
        try {
            const data = JSON.parse(e.data);
            if (data.msg) appendLog(data.msg);
        } catch (_) {}
    };
    logEventSource.onerror = function () {
        // EventSource 会根据服务端 retry 指令自动重连，无需创建重复连接。
        console.warn("实时日志连接暂时中断，等待自动重连");
    };
}

function stopLogStream() {
    if (logEventSource) {
        logEventSource.close();
        logEventSource = null;
    }
}

async function loadConfig() {
    try {
        const res = await fetch("/api/config");
        const cfg = await res.json();
        document.getElementById("cfgEnabled").checked = cfg.enabled;
        document.getElementById("cfgSmtpServer").value = cfg.smtp_server || "";
        document.getElementById("cfgSmtpPort").value = cfg.smtp_port || 587;
        document.getElementById("cfgSenderEmail").value = cfg.sender_email || "";
        document.getElementById("cfgReceiverEmail").value = cfg.receiver_email || "";
        document.getElementById("cfgCooldown").value = cfg.cooldown_seconds ?? 180;

        const envHint = document.getElementById("envHint");
        if (cfg.has_env_password) {
            envHint.style.display = "inline";
        } else {
            envHint.style.display = "none";
        }
    } catch (e) {
        console.error("加载配置失败", e);
    }
}

async function saveConfig() {
    const payload = {
        enabled: document.getElementById("cfgEnabled").checked,
        smtp_server: document.getElementById("cfgSmtpServer").value.trim(),
        smtp_port: parseInt(document.getElementById("cfgSmtpPort").value) || 587,
        sender_email: document.getElementById("cfgSenderEmail").value.trim(),
        receiver_email: document.getElementById("cfgReceiverEmail").value.trim(),
        cooldown_seconds: parseInt(document.getElementById("cfgCooldown").value) || 180,
    };

    const statusEl = document.getElementById("configStatus");
    statusEl.textContent = "保存中...";
    try {
        const res = await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.ok) {
            statusEl.textContent = "配置已保存";
            loadConfig();
        } else {
            statusEl.textContent = data.message || "保存失败";
        }
    } catch (e) {
        statusEl.textContent = "请求失败: " + e;
    }
    setTimeout(() => { statusEl.textContent = ""; }, 3000);
    return false;
}

async function testEmail() {
    const statusEl = document.getElementById("configStatus");
    statusEl.textContent = "正在发送测试邮件...";
    try {
        const res = await fetch("/api/config/test-email", { method: "POST" });
        const data = await res.json();
        statusEl.textContent = data.message || (data.ok ? "测试成功" : "测试失败");
    } catch (_) {
        statusEl.textContent = "测试请求失败";
    }
}

fetchStatus();
setInterval(fetchStatus, 3000);
loadFaces();
loadConfig();
loadCameraSelection();
loadEvents();
setInterval(loadEvents, 15000);

// 页面完成加载后再建立永久 SSE 连接，避免浏览器一直显示加载状态。
if (document.readyState === "complete") {
    startLogStream();
} else {
    window.addEventListener("load", startLogStream, { once: true });
}
window.addEventListener("beforeunload", stopLogStream);
