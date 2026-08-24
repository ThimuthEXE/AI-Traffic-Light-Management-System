// AI Traffic Light Management Dashboard - Interactive Client Logic

const API_BASE = window.location.origin;
const WS_URL = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/live-traffic`;

let socket = null;
let mlChart = null;
let benchmarkChart = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    initWebSocket();
    initCharts();
    loadAnalytics();
    
    // Set default ML datetime to tomorrow 8:00 AM
    const now = new Date();
    now.setDate(now.getDate() + 1);
    now.setHours(8, 0, 0, 0);
    document.getElementById("ml-datetime").value = now.toISOString().slice(0, 16);
    
    runMLPrediction();
});

// Tab Navigation Switcher
function switchTab(tabId) {
    document.getElementById("tab-live").classList.add("hidden");
    document.getElementById("tab-ml").classList.add("hidden");
    document.getElementById("tab-analytics").classList.add("hidden");

    document.querySelectorAll("[id^='tab-btn-']").forEach(btn => {
        btn.classList.remove("tab-active", "bg-emerald-500/10", "text-emerald-400", "border", "border-emerald-500/30");
        btn.classList.add("text-slate-400");
    });

    const activeContent = document.getElementById(`tab-${tabId}`);
    const activeBtn = document.getElementById(`tab-btn-${tabId}`);

    if (activeContent) activeContent.classList.remove("hidden");
    if (activeBtn) {
        activeBtn.classList.add("tab-active", "bg-emerald-500/10", "text-emerald-400", "border", "border-emerald-500/30");
        activeBtn.classList.remove("text-slate-400");
    }

    if (tabId === "analytics") {
        loadAnalytics();
    }
}

// WebSocket Connection & Real-Time Telemetry Stream
function initWebSocket() {
    const statusPill = document.getElementById("ws-status");
    const statusText = document.getElementById("ws-text");

    try {
        socket = new WebSocket(WS_URL);

        socket.onopen = () => {
            statusText.innerText = "Live Telemetry";
            statusPill.classList.remove("border-red-500/50");
            statusPill.classList.add("border-emerald-500/50");
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateLiveDashboard(data);
            } catch (e) {
                console.error("Error parsing telemetry frame:", e);
            }
        };

        socket.onclose = () => {
            statusText.innerText = "Reconnecting...";
            statusPill.classList.add("border-red-500/50");
            setTimeout(initWebSocket, 2000);
        };

        socket.onerror = () => {
            socket.close();
        };
    } catch (err) {
        console.error("WebSocket init failed:", err);
    }
}

// Update UI elements based on live telemetry snapshot
function updateLiveDashboard(data) {
    if (!data) return;

    const phase = data.active_phase || {};
    const countdown = phase.countdown_seconds !== undefined ? phase.countdown_seconds : "--";
    const state = phase.state || "GREEN";

    // Center Countdown & Phase
    const centerCountdown = document.getElementById("center-countdown");
    const centerState = document.getElementById("center-state-label");
    const phaseBadge = document.getElementById("active-phase-badge");

    centerCountdown.innerText = `${countdown}s`;
    centerState.innerText = state;
    
    if (state === "GREEN") {
        centerCountdown.className = "text-3xl font-black text-emerald-400 font-mono";
        centerState.className = "text-[10px] font-bold text-emerald-300";
    } else if (state === "YELLOW") {
        centerCountdown.className = "text-3xl font-black text-yellow-400 font-mono";
        centerState.className = "text-[10px] font-bold text-yellow-300";
    } else {
        centerCountdown.className = "text-3xl font-black text-red-400 font-mono";
        centerState.className = "text-[10px] font-bold text-red-300";
    }

    phaseBadge.innerText = `PHASE: ${phase.phase_name || "EW GREEN"}`;

    // Signal Heads Lenses
    const sigs = data.signals || {};
    updateSignalLenses("n", sigs.North || "RED");
    updateSignalLenses("s", sigs.South || "RED");
    updateSignalLenses("e", sigs.East || "GREEN");
    updateSignalLenses("w", sigs.West || "GREEN");

    // Allocated & Elapsed
    document.getElementById("val-allocated-green").innerText = `${phase.allocated_green || 20.0}s`;
    document.getElementById("val-elapsed-time").innerText = `${phase.elapsed_in_state || 0.0}s`;
    document.getElementById("val-controller-name").innerText = data.control_mode || "AI Cycle-Adaptive";

    // Approach Badges & Sliders
    const apps = data.approaches || {};
    if (apps.North) updateApproachBadge("n", apps.North);
    if (apps.South) updateApproachBadge("s", apps.South);
    if (apps.East) updateApproachBadge("e", apps.East);
    if (apps.West) updateApproachBadge("w", apps.West);

    // Decision Inspector
    const decision = data.ai_decision || {};
    document.getElementById("val-demand-share").innerText = `${decision.target_ratio || 50.0}%`;
    document.getElementById("val-next-green").innerText = `${decision.green_duration || 22.0}s`;
    document.getElementById("val-target-pcu").innerText = `${decision.target_pcu || 0.0} PCU`;
    document.getElementById("val-cross-pcu").innerText = `${decision.cross_pcu || 0.0} PCU`;

    // Metrics
    const metrics = data.metrics || {};
    document.getElementById("kpi-awt").innerText = `${metrics.average_wait_time || 4.8}s`;
    document.getElementById("kpi-cleared").innerText = metrics.cleared_vehicles || 0;
}

function updateSignalLenses(dir, state) {
    const red = document.getElementById(`sig-${dir}-red`);
    const yellow = document.getElementById(`sig-${dir}-yellow`);
    const green = document.getElementById(`sig-${dir}-green`);

    red.className = state === "RED" ? "w-4 h-4 rounded-full light-red" : "w-4 h-4 rounded-full light-red-off";
    yellow.className = state === "YELLOW" ? "w-4 h-4 rounded-full light-yellow" : "w-4 h-4 rounded-full light-yellow-off";
    green.className = state === "GREEN" ? "w-4 h-4 rounded-full light-green" : "w-4 h-4 rounded-full light-green-off";
}

function updateApproachBadge(dir, info) {
    const badge = document.getElementById(`badge-${dir}`);
    if (!badge) return;

    const vpm = Math.round(info.arrival_flow_vpm);
    const cat = info.density_category || "MED";
    
    badge.innerText = `${vpm} v/m [${cat}]`;
    if (cat === "HIGH") {
        badge.className = "px-1.5 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-400";
    } else if (cat === "MEDIUM") {
        badge.className = "px-1.5 py-0.5 rounded text-[10px] font-bold bg-yellow-500/20 text-yellow-400";
    } else {
        badge.className = "px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400";
    }
}

// Slider Change Handler
async function onDensitySliderChange(direction, value) {
    try {
        await fetch(`${API_BASE}/api/intersections/INT-KDU-01/density`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ direction: direction, spawn_interval_sec: parseFloat(value) })
        });
    } catch (e) {
        console.error("Failed to update density:", e);
    }
}

// Mode Switch Handler
async function setMode(mode) {
    const btnFixed = document.getElementById("mode-btn-fixed");
    const btnAi = document.getElementById("mode-btn-ai");

    if (mode === 1) {
        btnFixed.className = "px-3 py-1 text-xs font-semibold rounded bg-amber-500 text-dark-950 shadow";
        btnAi.className = "px-3 py-1 text-xs font-medium rounded text-slate-400 hover:text-white transition";
    } else {
        btnAi.className = "px-3 py-1 text-xs font-semibold rounded bg-emerald-500 text-dark-950 shadow";
        btnFixed.className = "px-3 py-1 text-xs font-medium rounded text-slate-400 hover:text-white transition";
    }

    try {
        await fetch(`${API_BASE}/api/intersections/INT-KDU-01/control`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: mode })
        });
    } catch (e) {
        console.error("Failed to switch mode:", e);
    }
}

// Emergency Preemption Trigger
async function triggerEmergencyOverride() {
    try {
        await fetch(`${API_BASE}/api/intersections/INT-KDU-01/control`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: 2, emergency_axis: "EW" })
        });
        alert("🚨 Emergency Preemption Signal Activated for Axis EW!");
    } catch (e) {
        console.error("Failed to trigger emergency:", e);
    }
}

// Machine Learning Forecasting
async function runMLPrediction() {
    const dtVal = document.getElementById("ml-datetime").value;
    const weatherVal = document.getElementById("ml-weather").value;

    try {
        const res = await fetch(`${API_BASE}/api/predictions/forecast?datetime_str=${encodeURIComponent(dtVal)}&weather=${encodeURIComponent(weatherVal)}`);
        const data = await res.json();

        // Update badge & timing recommendations
        document.getElementById("ml-critical-badge").innerText = `🚨 ${data.critical_tidal_approach} (Bottleneck)`;
        
        const timing = data.proactive_timing_recommendation || {};
        document.getElementById("ml-timing-ns").innerText = `${timing.Phase_NS_Green || 29.0}s`;
        document.getElementById("ml-timing-ew").innerText = `${timing.Phase_EW_Green || 23.0}s`;

        // Update Chart
        const flows = data.predicted_flows_vpm || { North: 35, South: 35, East: 30, West: 30 };
        if (mlChart) {
            mlChart.data.datasets[0].data = [flows.North, flows.South, flows.East, flows.West];
            mlChart.update();
        }
    } catch (e) {
        console.error("Prediction fetch failed:", e);
    }
}

// Historical Analytics Loader
async function loadAnalytics() {
    try {
        const res = await fetch(`${API_BASE}/api/analytics/cycles?limit=10`);
        const logs = await res.json();
        
        const tbody = document.getElementById("logs-table-body");
        if (!tbody) return;

        if (logs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center p-4 text-slate-500">No completed cycles recorded yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = logs.map(l => `
            <tr class="hover:bg-slate-800/40 transition">
                <td class="p-2.5 font-mono text-slate-400">${new Date(l.logged_at * 1000).toLocaleTimeString()}</td>
                <td class="p-2.5 font-bold ${l.phase_axis === 'NS' ? 'text-blue-400' : 'text-purple-400'}">Phase ${l.phase_axis}</td>
                <td class="p-2.5 font-bold text-yellow-400">${l.allocated_green}s</td>
                <td class="p-2.5 font-semibold text-emerald-400">${l.vehicles_cleared} veh</td>
                <td class="p-2.5 text-slate-300">${l.pcu_demand || '--'} PCU</td>
                <td class="p-2.5 text-slate-400">${l.max_wait_time || '0.0'}s</td>
            </tr>
        `).join("");
    } catch (e) {
        console.error("Analytics fetch failed:", e);
    }
}

// Chart.js Initialization
function initCharts() {
    // 1. ML Flow Chart
    const ctxML = document.getElementById("mlFlowChart");
    if (ctxML) {
        mlChart = new Chart(ctxML, {
            type: "bar",
            data: {
                labels: ["North Approach", "South Approach", "East Approach", "West Approach"],
                datasets: [{
                    label: "Predicted Flow (veh/min)",
                    data: [130, 20, 105, 18],
                    backgroundColor: [
                        "rgba(52, 152, 219, 0.75)",
                        "rgba(52, 152, 219, 0.4)",
                        "rgba(155, 89, 182, 0.75)",
                        "rgba(155, 89, 182, 0.4)"
                    ],
                    borderColor: [
                        "#3498db",
                        "#3498db",
                        "#9b59b6",
                        "#9b59b6"
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#cbd5e1", font: { weight: "600" } }
                    }
                }
            }
        });
    }

    // 2. Academic Benchmark Comparison Chart
    const ctxBench = document.getElementById("benchmarkChart");
    if (ctxBench) {
        benchmarkChart = new Chart(ctxBench, {
            type: "bar",
            data: {
                labels: ["Morning Rush", "Midday Commercial", "Evening Rush", "Off-Peak Night"],
                datasets: [
                    {
                        label: "Fixed-Time Baseline (H0)",
                        data: [28.4, 21.0, 31.2, 18.0],
                        backgroundColor: "rgba(239, 68, 68, 0.6)",
                        borderColor: "#ef4444",
                        borderWidth: 1.5,
                        borderRadius: 6
                    },
                    {
                        label: "AI Cycle-Adaptive (H1)",
                        data: [7.2, 4.5, 8.1, 2.8],
                        backgroundColor: "rgba(16, 185, 129, 0.7)",
                        borderColor: "#10b981",
                        borderWidth: 1.5,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: { color: "#e2e8f0", font: { size: 11, weight: "600" } }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: "Average Waiting Time (seconds)", color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#cbd5e1" }
                    }
                }
            }
        });
    }
}
