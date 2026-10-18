Recorded feed responses, fetched 2026-10-18 with user-agent `swimtemps/0.1`.

| File | Source | Shape |
|---|---|---|
| `clevedon_266367_168h_2026-10-18.json` | `https://clevedonmarinelake.org.uk/wp-json/cml/v1/temps?sensor=266367&hours=168` (deep) | columns: `labels` (epoch ms), `datasets[0].data` (°C), half-hourly |
| `clevedon_666332_168h_2026-10-18.json` | same, `sensor=666332` (shallow) | as above |
| `clevedon_air_2026-10-18.json` | `https://api.open-meteo.com/v1/forecast?latitude=51.435&longitude=-2.87&hourly=temperature_2m&timezone=UTC&past_days=8` | Open-Meteo hourly air temperature, UTC, 2026-10-10 to 2026-10-18, for the cleaning tests |
| `cromhall_lake_2026-10-18.json` | `https://script.google.com/macros/s/AKfycbxHEDCY3xTIqrM-AXF9YuPad2W5IYQZ-G-iaDBBMpozS7kDNjC65ELC1GbTx0ydyLSp/exec` (The Lake, Cromhall) | rows: header row, then `["d-Mon-YYYY", "°C as text"]` |

Clevedon's API serves at most 7 days, so these are the only copies of the
2026-10-18 fault: from 04:00 UTC both sensors drop ~4 °C in 30 minutes to the
overnight air temperature; the deep sensor recovers at 08:30, the shallow one
had not by 09:00. Cromhall's rows include a typo (`25..1`) and 20 repeated dates.
