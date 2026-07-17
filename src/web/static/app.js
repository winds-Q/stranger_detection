const logViewer = document.getElementById("logViewer");
const btnStart  = document.getElementById("btnStart");
const btnStop   = document.getElementById("btnStop");
const statusDot = document.getElementById("statusDot");
const statusTxt = document.getElementById("statusText");

let logLines = [];
let logEventSource = null;

function updateButtons(running) {
    btnStart.disabled = running;
    btnStop.disabled  = !running;
    statusDot.className = running ? "dot on" : "dot off";
    statusTxt.textContent = running ? "运行中" : "未运行";
}

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

function handleUpload(input) {
    const files = input.files;
    if (!files.length) return;

    const statusEl = document.getElementById("uploadStatus");
    statusEl.textContent = "上传中...";

    let done = 0;
    let failed = 0;
    for (const file of files) {
        const form = new FormData();
        form.append("file", file);

        fetch("/api/faces/upload", { method: "POST", body: form })
            .then(async r => ({ ok: r.ok, data: await r.json() }))
            .then(({ ok, data }) => {
                done++;
                if (!ok || !data.ok) {
                    failed++;
                }
                if (done === files.length && failed === 0) {
                    statusEl.textContent = `已上传 ${done} 个文件`;
                    loadFaces();
                    input.value = "";
                } else if (done === files.length) {
                    statusEl.textContent = `${done - failed} 个成功，${failed} 个失败`;
                    loadFaces();
                    input.value = "";
                }
            })
            .catch(() => {
                done++;
                failed++;
                if (done === files.length) {
                    statusEl.textContent = `${done - failed} 个成功，${failed} 个失败`;
                    loadFaces();
                    input.value = "";
                }
            });
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
                return `<div class="face-item">
                    <span class="name" title="${safeName}">${safeName}</span>
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
        document.getElementById("cfgSenderPassword").value = cfg.sender_password || "";
        document.getElementById("cfgReceiverEmail").value = cfg.receiver_email || "";
        document.getElementById("cfgCooldown").value = cfg.cooldown_seconds ?? 180;

        const envHint = document.getElementById("envHint");
        if (cfg.has_env_password) {
            envHint.style.display = "inline";
            document.getElementById("cfgSenderPassword").disabled = true;
        } else {
            envHint.style.display = "none";
            document.getElementById("cfgSenderPassword").disabled = false;
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
        sender_password: document.getElementById("cfgSenderPassword").value,
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

fetchStatus();
loadFaces();
loadConfig();

// 页面完成加载后再建立永久 SSE 连接，避免浏览器一直显示加载状态。
if (document.readyState === "complete") {
    startLogStream();
} else {
    window.addEventListener("load", startLogStream, { once: true });
}
window.addEventListener("beforeunload", stopLogStream);
