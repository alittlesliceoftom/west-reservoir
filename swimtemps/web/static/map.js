// Location step of the create wizard: search, a draggable pin, and a weather
// preview for the chosen spot so the user can see the weather looks right.
(function () {
  function ready(fn) { if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn); else fn(); }

  ready(function () {
    var form = document.getElementById("location-form");
    var cont = document.getElementById("continue");
    var preview = document.getElementById("preview");
    var nearbyBox = document.getElementById("nearby");
    var results = document.getElementById("results");
    var map = L.map("map").setView([54.0, -2.5], 5);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);
    var marker = null;
    var lookup = 0;

    function text(el, value) { el.textContent = value; el.hidden = !value; }

    function place(lat, lon, zoom) {
      lat = +(+lat).toFixed(5); lon = +(+lon).toFixed(5);
      if (!marker) {
        marker = L.marker([lat, lon], { draggable: true }).addTo(map);
        marker.on("dragend", function () { var p = marker.getLatLng(); place(p.lat, p.lng); });
      } else {
        marker.setLatLng([lat, lon]);
      }
      if (zoom) map.setView([lat, lon], zoom);
      form.lat.value = lat; form.lon.value = lon; form.timezone.value = "";
      cont.disabled = true;
      nearbyBox.hidden = true;
      text(preview, "Checking the weather for this spot...");
      var mine = ++lookup;
      fetch("/api/point?lat=" + lat + "&lon=" + lon, { headers: { Accept: "application/json" } })
        .then(function (r) { return r.json().then(function (body) { return { ok: r.ok, body: body }; }); })
        .then(function (res) {
          if (mine !== lookup) return;
          if (!res.ok) { text(preview, res.body.detail || "Could not check the weather here."); return; }
          var b = res.body;
          form.timezone.value = b.timezone;
          text(preview, "Weather for this spot: " + (+b.air_temp).toFixed(1) + " °C air now, elevation " +
            Math.round(b.elevation) + " m, timezone " + b.timezone + ".");
          if (b.nearby && b.nearby.length) {
            var w = b.nearby[0];
            nearbyBox.innerHTML = "";
            var p = document.createElement("p");
            p.textContent = "Is this " + w.name + "? It is already on swimtemps, " + w.distance_m + " m from your pin.";
            var yes = document.createElement("a");
            yes.href = "/w/" + w.slug; yes.className = "button primary"; yes.textContent = "Yes, go to " + w.name;
            var no = document.createElement("button");
            no.type = "button"; no.textContent = "No, it's a different water";
            no.onclick = function () { nearbyBox.hidden = true; cont.disabled = false; };
            var row = document.createElement("p"); row.style.display = "flex"; row.style.gap = "8px"; row.style.flexWrap = "wrap";
            row.appendChild(yes); row.appendChild(no);
            nearbyBox.appendChild(p); nearbyBox.appendChild(row);
            nearbyBox.hidden = false;
          } else {
            cont.disabled = false;
          }
        })
        .catch(function () { if (mine === lookup) text(preview, "Could not check the weather here. Try again."); });
    }

    map.on("click", function (e) { place(e.latlng.lat, e.latlng.lng); });

    document.getElementById("search").addEventListener("submit", function (e) {
      e.preventDefault();
      var q = e.target.q.value.trim();
      if (q.length < 2) return;
      results.innerHTML = "<li class='muted'>Searching...</li>";
      fetch("/api/geocode?q=" + encodeURIComponent(q))
        .then(function (r) { return r.json(); })
        .then(function (places) {
          results.innerHTML = "";
          if (!Array.isArray(places) || !places.length) { results.innerHTML = "<li class='muted'>No places found. Move the map by hand instead.</li>"; return; }
          places.forEach(function (pl) {
            var li = document.createElement("li");
            var b = document.createElement("button");
            b.type = "button";
            b.textContent = [pl.name, pl.region, pl.country].filter(Boolean).join(", ");
            b.onclick = function () { results.innerHTML = ""; place(pl.lat, pl.lon, 14); };
            li.appendChild(b); results.appendChild(li);
          });
        })
        .catch(function () { results.innerHTML = "<li class='muted'>Search failed. Move the map by hand instead.</li>"; });
    });

    document.getElementById("locate").addEventListener("click", function () {
      if (!navigator.geolocation) return;
      navigator.geolocation.getCurrentPosition(function (pos) { place(pos.coords.latitude, pos.coords.longitude, 15); });
    });
  });
})();
