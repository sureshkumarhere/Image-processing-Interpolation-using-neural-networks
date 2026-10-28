/**
 * Benchmark Analytics Dashboard - Dynamic Multi-View Visualization (Issue #10)
 * Single-column layout with rich zoom controls, dataset symbols, and interactive Plotly tools.
 */

document.addEventListener("DOMContentLoaded", () => {
  let allRecords = [];
  let currentMetric = "psnr"; // "psnr" | "ssim"
  let isLogX = false;
  let sortColumn = "mean_psnr";
  let sortAsc = false;

  // DOM Elements
  const filterDataset = document.getElementById("filter-dataset");
  const filterScale = document.getElementById("filter-scale");
  const filterCondition = document.getElementById("filter-condition");
  const toggleMetricPsnr = document.getElementById("toggle-metric-psnr");
  const toggleMetricSsim = document.getElementById("toggle-metric-ssim");
  const btnResetFilters = document.getElementById("btn-reset-filters");
  const btnExportCsv = document.getElementById("btn-export-csv");
  const btnToggleLogX = document.getElementById("btn-toggle-log-x");
  const tableSearch = document.getElementById("table-search");

  // KPI Elements
  const kpiBestPsnr = document.getElementById("kpi-best-psnr");
  const kpiBestPsnrSub = document.getElementById("kpi-best-psnr-sub");
  const kpiBestSsim = document.getElementById("kpi-best-ssim");
  const kpiBestSsimSub = document.getElementById("kpi-best-ssim-sub");
  const kpiFastestTime = document.getElementById("kpi-fastest-time");
  const kpiFastestTimeSub = document.getElementById("kpi-fastest-time-sub");
  const kpiTotalRuns = document.getElementById("kpi-total-runs");
  const kpiTotalRunsSub = document.getElementById("kpi-total-runs-sub");

  // Table Body & Headers
  const tableBody = document.querySelector("#analytics-table tbody");
  const tableHeaders = document.querySelectorAll("#analytics-table th[data-sort]");

  // Color Mapping
  const MODEL_COLORS = {
    our_nn: "#38bdf8",     // Sky Blue (Proposed)
    srcnn: "#f59e0b",      // Amber
    lanczos: "#a855f7",    // Purple
    bicubic: "#22c55e",    // Emerald
    bilinear: "#ec4899",   // Pink
    nearest: "#94a3b8"     // Slate Grey
  };

  const MODEL_LABELS = {
    our_nn: "OurNN (Proposed)",
    srcnn: "SRCNN",
    lanczos: "Lanczos",
    bicubic: "Bicubic",
    bilinear: "Bilinear",
    nearest: "Nearest"
  };

  const DATASET_SYMBOLS = {
    Set5: "circle",
    Set14: "square",
    BSD100: "diamond",
    Urban100: "triangle-up"
  };

  // Base Plotly Theme Layout for Dark Slate Theme
  function getBasePlotlyLayout(title, xTitle, yTitle, height = 520) {
    return {
      title: {
        text: `<b>${title}</b>`,
        font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size: 15, color: "#f8fafc" },
        x: 0.02,
        y: 0.95
      },
      hovermode: "closest",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(11, 15, 25, 0.7)",
      font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color: "#94a3b8", size: 11 },
      xaxis: {
        title: { text: xTitle, font: { size: 12, color: "#cbd5e1" } },
        gridcolor: "rgba(255, 255, 255, 0.06)",
        zerolinecolor: "rgba(255, 255, 255, 0.12)",
        color: "#94a3b8",
        tickfont: { color: "#94a3b8" }
      },
      yaxis: {
        title: { text: yTitle, font: { size: 12, color: "#cbd5e1" } },
        gridcolor: "rgba(255, 255, 255, 0.06)",
        zerolinecolor: "rgba(255, 255, 255, 0.12)",
        color: "#94a3b8",
        tickfont: { color: "#94a3b8" }
      },
      legend: {
        font: { color: "#e2e8f0", size: 11 },
        bgcolor: "rgba(15, 23, 42, 0.75)",
        bordercolor: "rgba(255, 255, 255, 0.1)",
        borderwidth: 1,
        orientation: "h",
        y: -0.18,
        x: 0.5,
        xanchor: "center"
      },
      margin: { l: 60, r: 28, t: 40, b: 68 },
      height: height,
      hoverlabel: {
        bgcolor: "#1e293b",
        bordercolor: "#38bdf8",
        font: { color: "#f8fafc", family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size: 12 }
      },
      autosize: true
    };
  }

  // Interactive Plotly Configuration with Zoom, ModeBar, and High-Res PNG
  const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    scrollZoom: true,
    displaylogo: false,
    modeBarButtonsToAdd: ["hoverclosest", "hovercompare"],
    toImageButtonOptions: {
      format: "png",
      filename: "benchmark_analytics_chart",
      height: 700,
      width: 1400,
      scale: 2
    }
  };

  // 1. Fetch Benchmark Data from API
  async function loadData() {
    try {
      const res = await fetch("/api/benchmark-summary");
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      allRecords = data.records || [];
      if (allRecords.length === 0) {
        showError("No benchmark records found in results/summary.csv.");
        return;
      }
      updateAll();
    } catch (err) {
      console.error("Failed to load benchmark data:", err);
      showError("Failed to fetch benchmark summary data. Ensure the server is running.");
    }
  }

  function showError(msg) {
    tableBody.innerHTML = `<tr><td colspan="9" class="text-center" style="color: #ef4444; padding: 2rem;">${msg}</td></tr>`;
  }

  // 2. Filter Records
  function getFilteredRecords() {
    const ds = filterDataset.value;
    const sc = filterScale.value;
    const cond = filterCondition.value;
    const search = (tableSearch.value || "").toLowerCase().trim();

    return allRecords.filter((r) => {
      // Exclude synthetic test_dummy from general charts unless explicitly picked
      if (ds === "all" && r.dataset === "test_dummy") return false;
      if (ds !== "all" && r.dataset !== ds) return false;

      if (sc !== "all" && r.scale !== parseInt(sc, 10)) return false;

      if (cond === "clean" && r.noise_sigma > 0) return false;
      if (cond === "noise15" && r.noise_sigma === 0) return false;

      if (search) {
        const str = `${r.experiment} ${r.model} ${r.dataset} ${r.scale} ${r.degradation}`.toLowerCase();
        if (!str.includes(search)) return false;
      }

      return true;
    });
  }

  // 3. Update KPI Cards
  function updateKPIs(records) {
    if (records.length === 0) {
      kpiBestPsnr.textContent = "--";
      kpiBestPsnrSub.textContent = "No data matching filters";
      kpiBestSsim.textContent = "--";
      kpiBestSsimSub.textContent = "No data matching filters";
      kpiFastestTime.textContent = "--";
      kpiFastestTimeSub.textContent = "No data matching filters";
      kpiTotalRuns.textContent = "0";
      kpiTotalRunsSub.textContent = "0 active runs";
      return;
    }

    // Best PSNR
    let bestP = records.reduce((max, r) => (r.mean_psnr > max.mean_psnr ? r : max), records[0]);
    kpiBestPsnr.textContent = `${bestP.mean_psnr.toFixed(2)} dB`;
    kpiBestPsnrSub.textContent = `${MODEL_LABELS[bestP.model] || bestP.model} (${bestP.dataset}, ×${bestP.scale})`;

    // Best SSIM
    let bestS = records.reduce((max, r) => (r.mean_ssim > max.mean_ssim ? r : max), records[0]);
    kpiBestSsim.textContent = bestS.mean_ssim.toFixed(4);
    kpiBestSsimSub.textContent = `${MODEL_LABELS[bestS.model] || bestS.model} (${bestS.dataset}, ×${bestS.scale})`;

    // Fastest Latency
    let fastest = records.reduce((min, r) => (r.mean_inference_time_s < min.mean_inference_time_s ? r : min), records[0]);
    const ms = fastest.mean_inference_time_s * 1000;
    kpiFastestTime.textContent = ms < 1 ? `${ms.toFixed(2)} ms` : `${ms.toFixed(1)} ms`;
    kpiFastestTimeSub.textContent = `${MODEL_LABELS[fastest.model] || fastest.model} (${fastest.dataset})`;

    // Total Filtered Runs
    kpiTotalRuns.textContent = records.length;
    kpiTotalRunsSub.textContent = `Across active filter selection`;
  }

  // 4. Chart 1: Performance vs Speed Trade-off
  function renderScatterChart(records) {
    const isPsnr = currentMetric === "psnr";
    const metricKey = isPsnr ? "mean_psnr" : "mean_ssim";
    const metricLabel = isPsnr ? "Mean PSNR (dB)" : "Mean SSIM";

    const traces = [];
    const models = [...new Set(records.map((r) => r.model))];

    models.forEach((m) => {
      const subset = records.filter((r) => r.model === m);
      const times = subset.map((r) => r.mean_inference_time_s * 1000);
      const metrics = subset.map((r) => r[metricKey]);
      const symbols = subset.map((r) => DATASET_SYMBOLS[r.dataset] || "circle");
      const sizes = subset.map((r) => {
        if (r.param_count === 0) return 13;
        const norm = Math.log10(Math.max(10, r.param_count));
        return Math.max(14, Math.min(34, norm * 5.2));
      });

      const customdata = subset.map((r) => [
        MODEL_LABELS[r.model] || r.model,
        r.dataset,
        r.scale,
        r.noise_sigma > 0 ? `Noise σ=${r.noise_sigma}` : "Clean",
        r.mean_psnr.toFixed(2),
        r.mean_ssim.toFixed(4),
        r.param_count,
        r.experiment
      ]);

      traces.push({
        name: MODEL_LABELS[m] || m,
        x: times,
        y: metrics,
        mode: "markers",
        marker: {
          size: sizes,
          symbol: symbols,
          color: MODEL_COLORS[m] || "#38bdf8",
          opacity: 0.85,
          line: { width: 1.5, color: "rgba(255,255,255,0.4)" }
        },
        customdata: customdata,
        hovertemplate:
          "<b>%{customdata[0]}</b> (%{customdata[7]})<br>" +
          "Dataset: %{customdata[1]} (×%{customdata[2]})<br>" +
          "Condition: %{customdata[3]}<br>" +
          "Latency: %{x:.2f} ms<br>" +
          "PSNR: %{customdata[4]} dB<br>" +
          "SSIM: %{customdata[5]}<br>" +
          "Params: %{customdata[6]:,}<extra></extra>"
      });
    });

    const layout = getBasePlotlyLayout(
      `1. Performance vs. Speed Trade-off (${isPsnr ? "PSNR" : "SSIM"} vs. Latency)`,
      "Inference Latency (ms)",
      metricLabel,
      520
    );

    if (isLogX) {
      layout.xaxis.type = "log";
      layout.xaxis.title.text = "Inference Latency (ms, log scale)";
    }

    Plotly.react("chart-scatter", traces, layout, PLOTLY_CONFIG);
  }

  // 5. Chart 2: Degradation & Robustness Analysis
  function renderBarChart(records) {
    const isPsnr = currentMetric === "psnr";
    const metricKey = isPsnr ? "mean_psnr" : "mean_ssim";
    const metricLabel = isPsnr ? "Mean PSNR (dB)" : "Mean SSIM";

    const datasets = [...new Set(records.map((r) => r.dataset))].sort();
    const models = [...new Set(records.map((r) => r.model))];

    const traces = models.map((m) => {
      const yValues = datasets.map((ds) => {
        const matches = records.filter((r) => r.model === m && r.dataset === ds);
        if (matches.length === 0) return null;
        const avg = matches.reduce((sum, r) => sum + r[metricKey], 0) / matches.length;
        return isPsnr ? parseFloat(avg.toFixed(2)) : parseFloat(avg.toFixed(4));
      });

      return {
        name: MODEL_LABELS[m] || m,
        x: datasets,
        y: yValues,
        type: "bar",
        marker: { color: MODEL_COLORS[m] || "#38bdf8" },
        hovertemplate: `<b>${MODEL_LABELS[m] || m}</b><br>Dataset: %{x}<br>${metricLabel}: %{y}<extra></extra>`
      };
    });

    const layout = getBasePlotlyLayout(
      `2. Robustness Analysis Across Scales & Noise Levels (${isPsnr ? "PSNR" : "SSIM"})`,
      "Benchmark Dataset",
      metricLabel,
      600
    );
    layout.barmode = "group";

    Plotly.react("chart-bar", traces, layout, PLOTLY_CONFIG);
  }

  // 6. Chart 3: Metric Distribution per Architecture
  function renderBoxChart(records) {
    const isPsnr = currentMetric === "psnr";
    const metricKey = isPsnr ? "mean_psnr" : "mean_ssim";
    const metricLabel = isPsnr ? "PSNR (dB)" : "SSIM";

    const models = [...new Set(records.map((r) => r.model))];
    const traces = models.map((m) => {
      const subset = records.filter((r) => r.model === m);
      const values = subset.map((r) => r[metricKey]);

      return {
        name: MODEL_LABELS[m] || m,
        y: values,
        type: "box",
        boxpoints: "all",
        jitter: 0.35,
        pointpos: -1.8,
        marker: {
          color: MODEL_COLORS[m] || "#38bdf8",
          size: 6,
          opacity: 0.8
        },
        line: { width: 1.5 },
        hovertemplate: `<b>${MODEL_LABELS[m] || m}</b><br>${metricLabel}: %{y}<extra></extra>`
      };
    });

    const layout = getBasePlotlyLayout(
      `3. SSIM / PSNR Quality Distribution per Model Architecture`,
      "Model Architecture Family",
      metricLabel,
      500
    );
    layout.showlegend = false;

    Plotly.react("chart-box", traces, layout, PLOTLY_CONFIG);
  }

  // 7. Chart 4: Parameter Efficiency Frontier
  function renderEfficiencyChart(records) {
    const isPsnr = currentMetric === "psnr";
    const metricKey = isPsnr ? "mean_psnr" : "mean_ssim";
    const metricLabel = isPsnr ? "PSNR (dB)" : "SSIM";

    const models = [...new Set(records.map((r) => r.model))];
    const traces = [];

    models.forEach((m) => {
      const subset = records.filter((r) => r.model === m);
      const params = subset.map((r) => Math.max(1, r.param_count)); // Avoid log(0)
      const metrics = subset.map((r) => r[metricKey]);
      const symbols = subset.map((r) => DATASET_SYMBOLS[r.dataset] || "circle");

      const customdata = subset.map((r) => [
        MODEL_LABELS[r.model] || r.model,
        r.dataset,
        r.scale,
        r.param_count,
        r.experiment,
        r[metricKey].toFixed(isPsnr ? 2 : 4)
      ]);

      traces.push({
        name: MODEL_LABELS[m] || m,
        x: params,
        y: metrics,
        mode: "markers",
        marker: {
          size: 14,
          symbol: symbols,
          color: MODEL_COLORS[m] || "#38bdf8",
          opacity: 0.85,
          line: { width: 1.5, color: "#1e293b" }
        },
        customdata: customdata,
        hovertemplate:
          "<b>%{customdata[0]}</b> (%{customdata[4]})<br>" +
          "Dataset: %{customdata[1]} (×%{customdata[2]})<br>" +
          "Parameters: %{customdata[3]:,}<br>" +
          `${metricLabel}: %{customdata[5]}<extra></extra>`
      });
    });

    const layout = getBasePlotlyLayout(
      `4. Parameter Efficiency Frontier (${isPsnr ? "PSNR" : "SSIM"} vs. Parameters)`,
      "Parameter Count (log scale)",
      metricLabel,
      520
    );
    layout.xaxis.type = "log";

    Plotly.react("chart-eff", traces, layout, PLOTLY_CONFIG);
  }

  // 8. Render Table
  function renderTable(records) {
    if (records.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="9" class="text-center text-muted" style="padding: 2rem;">No matching benchmark runs.</td></tr>`;
      return;
    }

    // Sort records
    const sorted = [...records].sort((a, b) => {
      let vA = a[sortColumn];
      let vB = b[sortColumn];
      if (typeof vA === "string") {
        return sortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
      }
      return sortAsc ? vA - vB : vB - vA;
    });

    // Best values for highlights
    const bestPsnr = Math.max(...records.map((r) => r.mean_psnr));
    const bestSsim = Math.max(...records.map((r) => r.mean_ssim));

    tableBody.innerHTML = "";
    sorted.forEach((r) => {
      const tr = document.createElement("tr");

      const isBestP = r.mean_psnr === bestPsnr;
      const isBestS = r.mean_ssim === bestSsim;

      const condBadge =
        r.noise_sigma > 0
          ? `<span class="badge badge-warning" style="font-size:0.72rem;">σ=${r.noise_sigma}</span>`
          : `<span class="badge badge-success" style="font-size:0.72rem;">Clean</span>`;

      tr.innerHTML = `
        <td style="font-weight: 500; font-family: monospace; font-size: 0.8rem; color: #cbd5e1;">${r.experiment}</td>
        <td>
          <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${MODEL_COLORS[r.model] || '#38bdf8'}; margin-right:6px;"></span>
          ${MODEL_LABELS[r.model] || r.model}
        </td>
        <td><span class="badge" style="font-size:0.75rem;">${r.dataset}</span></td>
        <td>×${r.scale}</td>
        <td>${condBadge}</td>
        <td>${r.param_count ? (r.param_count / 1000).toFixed(1) + "k" : "0"}</td>
        <td>${(r.mean_inference_time_s * 1000).toFixed(1)} ms</td>
        <td class="${isBestP ? "best-val" : ""}" style="font-weight:600;">${r.mean_psnr.toFixed(2)} dB</td>
        <td class="${isBestS ? "best-val" : ""}" style="font-weight:600;">${r.mean_ssim.toFixed(4)}</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // 9. Export Filtered Records to CSV
  function exportCSV() {
    const records = getFilteredRecords();
    if (records.length === 0) {
      alert("No records to export.");
      return;
    }

    const headers = [
      "experiment",
      "model",
      "dataset",
      "scale",
      "degradation",
      "noise_sigma",
      "n_images",
      "param_count",
      "mean_inference_time_s",
      "mean_psnr",
      "mean_ssim"
    ];

    let csvContent = headers.join(",") + "\n";
    records.forEach((r) => {
      const row = headers.map((h) => {
        const val = r[h] !== undefined ? r[h] : "";
        return typeof val === "string" && val.includes(",") ? `"${val}"` : val;
      });
      csvContent += row.join(",") + "\n";
    });

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `super_resolution_benchmark_${currentMetric}_filtered.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  // 10. Zoom / Reset / PNG Action Handlers
  function zoomChart(chartId, factor = 0.75) {
    const el = document.getElementById(chartId);
    if (!el || !el.layout) return;
    const xRange = el.layout.xaxis ? el.layout.xaxis.range : null;
    const yRange = el.layout.yaxis ? el.layout.yaxis.range : null;
    if (!xRange || !yRange) return;

    const xMid = (xRange[0] + xRange[1]) / 2;
    const xSpan = ((xRange[1] - xRange[0]) * factor) / 2;
    const yMid = (yRange[0] + yRange[1]) / 2;
    const ySpan = ((yRange[1] - yRange[0]) * factor) / 2;

    Plotly.relayout(el, {
      "xaxis.range": [xMid - xSpan, xMid + xSpan],
      "yaxis.range": [yMid - ySpan, yMid + ySpan]
    });
  }

  function resetChart(chartId) {
    const el = document.getElementById(chartId);
    if (!el) return;
    Plotly.relayout(el, {
      "xaxis.autorange": true,
      "yaxis.autorange": true
    });
  }

  function downloadChartPNG(chartId, filename) {
    Plotly.downloadImage(chartId, { format: "png", width: 1400, height: 750, filename: filename });
  }

  // Bind zoom in/out/reset/png buttons
  document.querySelectorAll(".btn-chart-zoom-in").forEach((btn) => {
    btn.addEventListener("click", () => zoomChart(btn.dataset.chart, 0.7));
  });
  document.querySelectorAll(".btn-chart-zoom-out").forEach((btn) => {
    btn.addEventListener("click", () => zoomChart(btn.dataset.chart, 1.4));
  });
  document.querySelectorAll(".btn-chart-reset").forEach((btn) => {
    btn.addEventListener("click", () => resetChart(btn.dataset.chart));
  });
  document.querySelectorAll(".btn-chart-png").forEach((btn) => {
    btn.addEventListener("click", () => downloadChartPNG(btn.dataset.chart, btn.dataset.name || "chart"));
  });

  // 11. Update Everything
  function updateAll() {
    const filtered = getFilteredRecords();
    updateKPIs(filtered);
    renderScatterChart(filtered);
    renderBarChart(filtered);
    renderBoxChart(filtered);
    renderEfficiencyChart(filtered);
    renderTable(filtered);
  }

  // Event Listeners for Filters
  [filterDataset, filterScale, filterCondition].forEach((el) => {
    el.addEventListener("change", updateAll);
  });

  tableSearch.addEventListener("input", () => {
    const filtered = getFilteredRecords();
    renderTable(filtered);
  });

  // Metric Toggle
  toggleMetricPsnr.addEventListener("click", () => {
    if (currentMetric === "psnr") return;
    currentMetric = "psnr";
    toggleMetricPsnr.classList.add("active");
    toggleMetricSsim.classList.remove("active");
    updateAll();
  });

  toggleMetricSsim.addEventListener("click", () => {
    if (currentMetric === "ssim") return;
    currentMetric = "ssim";
    toggleMetricSsim.classList.add("active");
    toggleMetricPsnr.classList.remove("active");
    updateAll();
  });

  // Log X Toggle
  btnToggleLogX.addEventListener("click", () => {
    isLogX = !isLogX;
    btnToggleLogX.classList.toggle("active", isLogX);
    btnToggleLogX.textContent = isLogX ? "Linear X" : "Log X";
    renderScatterChart(getFilteredRecords());
  });

  // Reset Filters
  btnResetFilters.addEventListener("click", () => {
    filterDataset.value = "all";
    filterScale.value = "all";
    filterCondition.value = "all";
    tableSearch.value = "";
    currentMetric = "psnr";
    toggleMetricPsnr.classList.add("active");
    toggleMetricSsim.classList.remove("active");
    isLogX = false;
    btnToggleLogX.classList.remove("active");
    btnToggleLogX.textContent = "Log X";
    updateAll();
  });

  // Export CSV
  btnExportCsv.addEventListener("click", exportCSV);

  // Table Sorting
  tableHeaders.forEach((th) => {
    th.style.cursor = "pointer";
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (sortColumn === col) {
        sortAsc = !sortAsc;
      } else {
        sortColumn = col;
        sortAsc = false;
      }
      tableHeaders.forEach((h) => (h.textContent = h.textContent.replace(/ [▲▼⬍]/, "") + " ⬍"));
      th.textContent = th.textContent.replace(" ⬍", "") + (sortAsc ? " ▲" : " ▼");
      renderTable(getFilteredRecords());
    });
  });

  // Window resize handler for responsive plotly charts
  window.addEventListener("resize", () => {
    ["chart-scatter", "chart-bar", "chart-box", "chart-eff"].forEach((id) => {
      const el = document.getElementById(id);
      if (el && el.data) {
        Plotly.Plots.resize(el);
      }
    });
  });

  // Initial Load
  loadData();
});
