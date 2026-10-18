// Renders the water page chart from the JSON embedded in the page.
// Times are shown in the water's own timezone, whatever the viewer's is.
(function () {
  function draw() {
    var el = document.getElementById("chart");
    var dataEl = document.getElementById("chart-data");
    if (!el || !dataEl || !window.echarts) return;
    var d = JSON.parse(dataEl.textContent);
    var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    var fmt = new Intl.DateTimeFormat("en-GB", { timeZone: d.timezone, weekday: "short", day: "numeric", month: "short" });
    var fmtFull = new Intl.DateTimeFormat("en-GB", { timeZone: d.timezone, weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    var t = function (iso) { return new Date(iso).getTime(); };

    var anchor = [t(d.latest_reading.at), d.latest_reading.temp];
    var points = d.forecast.map(function (f) { return { x: t(f.at), temp: f.temp, lower: f.lower, upper: f.upper }; });
    var lower = [[anchor[0], anchor[1]]].concat(points.map(function (p) { return [p.x, p.lower]; }));
    var width = [[anchor[0], 0]].concat(points.map(function (p) { return [p.x, +(p.upper - p.lower).toFixed(2)]; }));
    var nowMs = t(d.now);
    var start = nowMs - 30 * 24 * 3600 * 1000;

    var chart = echarts.init(el, dark ? "dark" : null, { renderer: "canvas" });
    chart.setOption({
      backgroundColor: "transparent",
      animation: false,
      grid: { left: 40, right: 12, top: 28, bottom: 56 },
      legend: { top: 0, data: ["Readings", "Forecast", "Air"], textStyle: { fontSize: 11 } },
      tooltip: {
        trigger: "axis",
        valueFormatter: function (v) { return v == null ? "" : (+v).toFixed(1) + " °C"; },
        axisPointer: { label: { formatter: function (p) { return fmtFull.format(new Date(p.value)); } } }
      },
      xAxis: { type: "time", axisLabel: { formatter: function (v) { return fmt.format(new Date(v)); }, hideOverlap: true } },
      yAxis: { type: "value", scale: true, name: "°C", nameGap: 8 },
      dataZoom: [{ type: "inside", startValue: start }, { type: "slider", startValue: start, height: 18, bottom: 8 }],
      series: [
        { name: "Air", type: "line", data: d.air, showSymbol: false, lineStyle: { width: 1, opacity: 0.35 }, itemStyle: { color: "#8a9aa6" }, z: 1 },
        { name: "band-lower", type: "line", data: lower, stack: "band", showSymbol: false, lineStyle: { opacity: 0 }, tooltip: { show: false }, silent: true },
        { name: "80% range", type: "line", data: width, stack: "band", showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: "#0b6e99", opacity: 0.18 }, tooltip: { show: false }, silent: true },
        { name: "Forecast", type: "line", data: d.trajectory, showSymbol: false, lineStyle: { type: "dashed", width: 2 }, itemStyle: { color: "#0b6e99" },
          markLine: { silent: true, symbol: "none", label: { formatter: "now", position: "start" }, lineStyle: { color: "#e3b341" }, data: [{ xAxis: nowMs }] } },
        { name: "Readings", type: "scatter", data: d.readings, symbolSize: 7, itemStyle: { color: dark ? "#e6edf2" : "#13202b" }, z: 3 }
      ]
    });
    window.addEventListener("resize", function () { chart.resize(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", draw); else draw();
})();
