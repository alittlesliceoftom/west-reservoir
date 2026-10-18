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

    // Readings are [time, °C, reason, note]: reason is null for a usable
    // reading and says why for one the cleaning flagged or an admin marked
    // wrong; note says when an admin accepted a flagged one.
    var usable = d.readings.filter(function (r) { return r[2] == null && r[3] == null; }).map(function (r) { return [r[0], r[1]]; });
    var accepted = d.readings.filter(function (r) { return r[2] == null && r[3] != null; }).map(function (r) { return [r[0], r[1], r[3]]; });
    var flagged = d.readings.filter(function (r) { return r[2] != null; });
    var points = d.forecast.map(function (f) { return { x: t(f.at), temp: f.temp, lower: f.lower, upper: f.upper }; });
    var lower = [], width = [];
    if (d.latest_reading && points.length) {
      var anchor = [t(d.latest_reading.at), d.latest_reading.temp];
      lower = [[anchor[0], anchor[1]]].concat(points.map(function (p) { return [p.x, p.lower]; }));
      width = [[anchor[0], 0]].concat(points.map(function (p) { return [p.x, +(p.upper - p.lower).toFixed(2)]; }));
    }
    var esc = function (s) { return window.echarts.format.encodeHTML(String(s)); };
    var nowMs = t(d.now);
    var start = nowMs - 30 * 24 * 3600 * 1000;

    var chart = echarts.init(el, dark ? "dark" : null, { renderer: "canvas" });
    chart.setOption({
      backgroundColor: "transparent",
      animation: false,
      grid: { left: 40, right: 12, top: 28, bottom: 56 },
      legend: { top: 0, data: ["Readings"].concat(accepted.length ? ["Accepted"] : [], flagged.length ? ["Flagged"] : [], ["Forecast", "Air"]), textStyle: { fontSize: 11 } },
      tooltip: {
        trigger: "axis",
        axisPointer: { label: { formatter: function (p) { return fmtFull.format(new Date(p.value)); } } },
        formatter: function (items) {
          var lines = [esc(fmtFull.format(new Date(items[0].axisValue)))];
          items.forEach(function (it) {
            if (it.seriesName.indexOf("band") === 0 || it.seriesName === "80% range") return;
            var v = Array.isArray(it.value) ? it.value[1] : it.value;
            if (v == null) return;
            var line = it.marker + esc(it.seriesName) + " " + (+v).toFixed(1) + " °C";
            if (it.seriesName === "Flagged" || it.seriesName === "Accepted") line += "<br><span style=\"opacity:.8\">" + esc(it.value[2]) + "</span>";
            lines.push(line);
          });
          return lines.join("<br>");
        }
      },
      xAxis: { type: "time", axisLabel: { formatter: function (v) { return fmt.format(new Date(v)); }, hideOverlap: true } },
      yAxis: { type: "value", scale: true, name: "°C", nameGap: 8 },
      dataZoom: [{ type: "inside", startValue: start }, { type: "slider", startValue: start, height: 18, bottom: 8 }],
      series: [
        { name: "Air", type: "line", data: d.air, showSymbol: false, lineStyle: { width: 1, opacity: 0.35 }, itemStyle: { color: "#8a9aa6" }, z: 1 },
        { name: "band-lower", type: "line", data: lower, stack: "band", showSymbol: false, lineStyle: { opacity: 0 }, tooltip: { show: false }, silent: true },
        { name: "80% range", type: "line", data: width, stack: "band", showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: "#0b6e99", opacity: 0.18 }, tooltip: { show: false }, silent: true },
        { name: "Forecast", type: "line", data: d.trajectory, showSymbol: false, lineStyle: { type: "dashed", width: 2 }, itemStyle: { color: "#0b6e99" },
          markLine: { silent: true, symbol: "none", label: { formatter: "now", position: "insideEndTop" }, lineStyle: { color: "#e3b341" }, data: [{ xAxis: nowMs }] } },
        { name: "Flagged", type: "scatter", data: flagged, symbolSize: 7, itemStyle: { color: "#9aa5ad", opacity: 0.7 }, z: 2 },
        { name: "Accepted", type: "scatter", data: accepted, symbol: "emptyCircle", symbolSize: 8, itemStyle: { color: dark ? "#e6edf2" : "#13202b", borderWidth: 2 }, z: 3 },
        { name: "Readings", type: "scatter", data: usable, symbolSize: 7, itemStyle: { color: dark ? "#e6edf2" : "#13202b" }, z: 3 }
      ]
    });
    window.addEventListener("resize", function () { chart.resize(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", draw); else draw();
})();
