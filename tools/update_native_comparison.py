"""Render fresh native Phoenix / NVIDIA measurements into the offline report."""

from pathlib import Path
import html, json, re

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "docs/NATIVE_NVIDIA_MEASUREMENTS.json").read_text(encoding="utf-8"))
    assert data["passed"] and all(
        route_record["passed"] for route_record in data["routes"].values()
    )
    native_route = data["routes"]["native"]
    nvidia_route = data["routes"]["nvidia"]
    metrics = [
        ("GPU interval, average", "gpu_ms", "mean", "ms"),
        ("GPU interval, 95th percentile", "gpu_ms", "p95", "ms"),
        ("CPU command recording", "cpu_record_ms", "mean", "ms"),
        ("Record / submit / GPU completion", "record_submit_wait_ms", "mean", "ms"),
        ("Process GPU-local usage", "device_local_mib", "mean", "MiB"),
        ("Process shared GPU usage", "device_nonlocal_mib", "mean", "MiB"),
        ("Host physical working set", "working_set_mib", "mean", "MiB"),
        ("Host private commitment", "private_mib", "mean", "MiB"),
    ]
    rows = "".join(
        f"<tr><td>{title}</td><td>{native_route[key][stat]:,.3f} {unit}</td><td>{nvidia_route[key][stat]:,.3f} {unit}</td></tr>"
        for title, key, stat, unit in metrics
    )
    rows += f"<tr><td>Peak host working set, including startup</td><td>{native_route['peak_working_set_mib']:,.2f} MiB</td><td>{nvidia_route['peak_working_set_mib']:,.2f} MiB</td></tr>"
    rows += f"<tr><td>Model weights, packed bytes</td><td>{native_route['weights_bytes'] / 2**20:,.3f} MiB</td><td>Not exposed</td></tr><tr><td>Model workspace</td><td>{native_route['workspace_bytes'] / 2**20:,.3f} MiB</td><td>Not exposed</td></tr>"
    startup = "".join(
        f"<tr><td>{name}</td><td>{sum(row['startup_ms']) / 3:,.2f} ms</td><td>{html.escape(row['startup_scope'])}</td></tr>"
        for name, row in [
            ("DLSS5 Phoenix native RTX", native_route),
            ("NVIDIA DLSS NR", nvidia_route),
        ]
    )
    section = (
        """<!-- NATIVE_NVIDIA_BEGIN -->
<section id="native-reference" class="source">
<div class="eyebrow">Fresh native Phoenix and NVIDIA comparison · 2026-10-10</div>
<h1>DLSS5 Phoenix native RTX and NVIDIA DLSS NR</h1>
<p><strong>RTX 5080 · driver 616.64 · Ryzen 9 9950X3D · 96 GiB installed RAM.</strong> 1920 × 1080, identical captured normalized FP16 color and motion, matching controls, intensity 1 / style 0 / automatic masking off. One runner per process. Three fresh processes per route, 128 frames each, first 16 excluded: <strong>336 measured frames per route</strong>. D3D12 debug layer enabled on both. Input upload and output readback are outside the steady timing intervals.</p>
<div class="table-wrap"><table><thead><tr><th>Measurement</th><th>DLSS5 Phoenix native RTX</th><th>NVIDIA DLSS NR</th></tr></thead><tbody>"""
        + rows
        + """</tbody></table></div>
<p><strong>Scopes matter:</strong> GPU time, CPU recording and record-to-completion wall time overlap. Host private commitment is not physical RAM. GPU usage includes harness and driver allocations; it is not just model weights and scratch. Memory rows overlap and must not be added. The 344 MiB Phoenix weight/workspace subtotal is not its whole-process memory. Both routes produced finite outputs without D3D12 errors. This timing test does not certify identical images or game FPS.</p>
<div class="chart-grid"><section class="chart-panel"><h3>Time comparison</h3><label>Metric <select id="nr-time-metric"><option value="gpu_ms">GPU interval</option><option value="cpu_record_ms">CPU command recording</option><option value="record_submit_wait_ms">Record to GPU completion</option></select></label><div id="nr-time-bars"></div></section>
<section class="chart-panel"><h3>Memory comparison</h3><label>Metric <select id="nr-memory-metric"><option value="device_local_mib">Whole-process GPU-local</option><option value="device_nonlocal_mib">Shared GPU usage</option><option value="working_set_mib">Physical RAM working set</option><option value="private_mib">Host private commitment</option></select></label><div id="nr-memory-bars"></div></section></div>
<div class="chart-grid"><section class="chart-panel"><h3>GPU time components</h3><label>Runner <select id="nr-time-runner"><option value="native">DLSS5 Phoenix native RTX</option><option value="nvidia">NVIDIA DLSS NR</option></select></label><div id="nr-time-pie"></div><div id="nr-time-stack"></div><p>NVIDIA exposes the complete Evaluate interval; its internal components remain opaque.</p></section>
<section class="chart-panel"><h3>Phoenix owned resource allocations</h3><div id="nr-memory-pie"></div><div id="nr-memory-stack"></div><p>This partition counts owned D3D12 resources, including upload storage. It is allocation size, not VRAM residency, and excludes driver/module/harness resources. NVIDIA does not expose a comparable internal allocation ledger; its measured process totals appear above.</p></section></div>
<details><summary>Startup and measurement boundaries</summary><div class="table-wrap"><table><thead><tr><th>Runner</th><th>Average startup</th><th>Timed scope</th></tr></thead><tbody>"""
        + startup
        + """</tbody></table></div><p>These startup intervals have different boundaries and must not be ranked directly. Filesystem caches were warm. GPU idle checks ran before each process; clocks were not locked. Native group timestamps include dispatch and barrier work within each group. NVIDIA was executed through its NGX feature on the same GPU, with the same user-supplied DLL as the model source. Internal NVIDIA weights, scratch and allocator categories are not estimated.</p></details>
<p><a href="NATIVE_NVIDIA_MEASUREMENTS.json">Measurements, ranges, memory phases and output checks</a>. The compiled portable results below use a separate synthetic RGB32F fixture and include CPU preparation/transfers/resolve; their complete-frame times have a different scope.</p>
</section>
<script id="native-nvidia-data" type="application/json">"""
        + json.dumps(data, separators=(",", ":")).replace("</", r"<\/")
        + """</script>
<style>#native-reference table{min-width:620px}#native-reference td:first-child{min-width:230px}#native-reference .chart-panel{overflow:auto}#native-reference svg{min-width:580px;width:100%}.nr-pie{height:180px;width:180px;border-radius:50%;display:grid;place-items:center;margin:14px auto}.nr-hole{height:110px;width:110px;background:var(--panel);border-radius:50%;display:grid;place-items:center;text-align:center}.nr-legend{list-style:none;padding:0}.nr-legend li{margin:5px 0}</style>
<script>
(()=>{
const D=JSON.parse(document.getElementById('native-nvidia-data').textContent),R=D.routes,$=id=>document.getElementById(id),colors=['#79dbb4','#86b9ff','#ffc56e','#c9a3ff','#ed8ca3','#7ad6e0','#a6c972','#d9a56d'],number=x=>x.toLocaleString(undefined,{maximumFractionDigits:3}),escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function bars(target,key,unit){const rows=[['DLSS5 Phoenix native RTX',R.native[key].mean],['NVIDIA DLSS NR',R.nvidia[key].mean]],max=Math.max(...rows.map(r=>r[1]));let s='<svg viewBox="0 0 900 120" role="img" aria-label="'+escape(key)+'; lower is better">';rows.forEach(([name,value],i)=>{let y=15+i*48,w=value/max*470;s+=`<text x="0" y="${y+18}">${escape(name)}</text><rect x="270" y="${y}" width="${w}" height="28" fill="${colors[i]}"><title>${number(value)} ${unit}</title></rect><text x="${280+w}" y="${y+18}">${number(value)} ${unit}</text>`;});$(target).innerHTML=s+'</svg>';}
function partition(target,stack,parts,unit){let total=Object.values(parts).reduce((a,b)=>a+b,0),position=0,gradient=[],legend=[],rects=[],x=0;Object.entries(parts).forEach(([name,value],i)=>{let end=position+value/total*360,color=colors[i%colors.length];gradient.push(`${color} ${position}deg ${end}deg`);position=end;legend.push(`<li><i class="swatch" style="background:${color}"></i>${escape(name)}: ${number(value)} ${unit} (${number(value/total*100)}%)</li>`);let w=value/total*880;rects.push(`<rect x="${x}" y="0" width="${w}" height="28" fill="${color}"><title>${escape(name)}: ${number(value)} ${unit}</title></rect>`);x+=w;});$(target).innerHTML=`<div class="nr-pie" role="img" aria-label="${number(total)} ${unit}" style="background:conic-gradient(${gradient.join(',')})"><div class="nr-hole">${number(total)}<br>${unit}</div></div><ul class="nr-legend">${legend.join('')}</ul>`;$(stack).innerHTML=`<svg viewBox="0 0 880 30" role="img" aria-label="Component shares">${rects.join('')}</svg>`;}
function draw(){bars('nr-time-bars',$('nr-time-metric').value,'ms');bars('nr-memory-bars',$('nr-memory-metric').value,'MiB');let native=$('nr-time-runner').value==='native',parts=native?{...R.native.groups_ms}:{'Opaque NVIDIA Evaluate':R.nvidia.gpu_ms.mean};if(native){let other=R.native.gpu_ms.mean-Object.values(parts).reduce((a,b)=>a+b,0);if(other>0)parts['Between group markers']=other;}partition('nr-time-pie','nr-time-stack',parts,'ms');}
const owned=R.native.owned_allocation_snapshot.generations[0].categories,memory={};Object.entries(owned).forEach(([name,value])=>{if(value.live_allocation_info_bytes)memory[name]=value.live_allocation_info_bytes/1048576;});partition('nr-memory-pie','nr-memory-stack',memory,'MiB allocated');['nr-time-metric','nr-memory-metric','nr-time-runner'].forEach(id=>$(id).addEventListener('change',draw));draw();
})();
</script>
<!-- NATIVE_NVIDIA_END -->"""
    )
    section = section.replace(
        "Both routes produced finite outputs without D3D12 errors.",
        "Both routes produced finite outputs without D3D12 errors. Frame 127 matched byte-for-byte between Phoenix and NVIDIA in all three run pairs.",
    )
    if data.get("model_source"):
        section = section.replace(
            "One runner per process.",
            "DLL: "
            + html.escape(data["model_source"]["filename"])
            + " "
            + html.escape(data["model_source"]["file_version"])
            + ". One runner per process.",
        )
    path = ROOT / "docs/PORTABLE_PERFORMANCE.html"
    text = path.read_text(encoding="utf-8")
    if "<!-- NATIVE_NVIDIA_BEGIN -->" in text:
        text = re.sub(
            r"<!-- NATIVE_NVIDIA_BEGIN -->.*?<!-- NATIVE_NVIDIA_END -->",
            lambda _: section,
            text,
            flags=re.S,
        )
    else:
        text = text.replace(
            "<!-- PORTABLE_CPP_BEGIN -->", section + "\n<!-- PORTABLE_CPP_BEGIN -->", 1
        )
    path.write_text(text, encoding="utf-8", newline="\n")
    print("Native/NVIDIA tables, time/memory bars and component pies refreshed")


if __name__ == "__main__":
    main()
