"""Render the fresh compiled-portable comparison into the existing offline report."""

from pathlib import Path
import argparse, html, json, re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/PORTABLE_CPP_PERFORMANCE.json")
    parser.add_argument("--html", type=Path, default=ROOT / "docs/PORTABLE_PERFORMANCE.html")
    args = parser.parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    if not data["passed"] or len(data["backends"]) != 6:
        raise ValueError("Complete validated six-route report required")
    esc = html.escape
    rows = []
    for backend_record in sorted(
        data["backends"], key=lambda backend_record: backend_record["operator_ms"]
    ):
        rows.append(
            "<tr><td><strong>"
            + esc(backend_record["label"])
            + '</strong><br><span class="small">'
            + esc(backend_record["hardware"] + " · " + backend_record["engine"])
            + "</span></td>"
            + "".join(
                "<td>" + cell_text + "</td>"
                for cell_text in [
                    f"{backend_record['operator_ms']:,.2f}",
                    f"{backend_record['frame_ms']:,.2f}",
                    f"{backend_record['graph_bytes'] / 2**20:,.2f}",
                    f"{backend_record['host_private_bytes'] / 2**20:,.2f}",
                    f"{backend_record['host_working_set_bytes'] / 2**20:,.2f}",
                    str(backend_record["samples"]),
                    f"{backend_record['frame_min_ms']:,.2f}–{backend_record['frame_max_ms']:,.2f}",
                ]
            )
            + "</tr>"
        )
    for backend_record in reversed(data["references"]):
        rows.insert(
            0,
            "<tr><td><strong>"
            + esc(backend_record["label"])
            + '</strong><br><span class="small">RTX 5080 · captured-input D3D12 reference</span></td>'
            + "".join(
                "<td>" + cell_text + "</td>"
                for cell_text in [
                    f"{backend_record['operator_ms']:,.4f}",
                    "—",
                    "—",
                    f"{backend_record['host_private_bytes'] / 2**20:,.2f}",
                    f"{backend_record['host_working_set_bytes'] / 2**20:,.2f}",
                    str(backend_record["samples"]),
                    "—",
                ]
            )
            + "</tr>",
        )
    changes = ""
    if data.get("before_after"):
        labels = {
            backend_record["id"]: backend_record["label"] for backend_record in data["backends"]
        }
        changes = (
            '<h3>Before and after this optimization pass</h3><div class="table-wrap"><table><thead><tr><th>Route</th><th>Before frame ms</th><th>After frame ms</th><th>Speedup</th><th>Before graph MiB</th><th>After graph MiB</th><th>Before host private MiB</th><th>After host private MiB</th></tr></thead><tbody>'
            + "".join(
                "<tr><td>"
                + esc(labels[backend_record["id"]])
                + "</td>"
                + "".join(
                    "<td>" + cell_text + "</td>"
                    for cell_text in [
                        f"{backend_record['before_frame_ms']:,.2f}",
                        f"{backend_record['after_frame_ms']:,.2f}",
                        f"{backend_record['speedup']:.2f}×",
                        f"{backend_record['before_graph_bytes'] / 2**20:,.2f}",
                        f"{backend_record['after_graph_bytes'] / 2**20:,.2f}",
                        f"{backend_record['before_host_private_bytes'] / 2**20:,.2f}",
                        f"{backend_record['after_host_private_bytes'] / 2**20:,.2f}",
                    ]
                )
                + "</tr>"
                for backend_record in data["before_after"]
            )
            + "</tbody></table></div>"
        )
    references = "".join(
        "<tr><td>"
        + esc(reference_record["label"])
        + "</td><td>"
        + f"{reference_record['operator_ms']:.4f}"
        + "</td><td>"
        + esc(reference_record["scope"])
        + "</td></tr>"
        for reference_record in data["references"]
    )
    thread_changes = ""
    if data.get("thread_comparison"):
        thread_rows = []
        for backend_record in data["thread_comparison"]:
            cells = [
                str(backend_record["before_threads"]),
                str(backend_record["after_threads"]),
                f"{backend_record['before_frame_ms']:,.2f}",
                f"{backend_record['after_frame_ms']:,.2f}",
                f"{backend_record['speedup']:.2f}×",
                f"{backend_record['before_host_private_bytes'] / 2**20:,.2f}",
                f"{backend_record['after_host_private_bytes'] / 2**20:,.2f}",
                f"{backend_record['before_host_working_set_bytes'] / 2**20:,.2f}",
                f"{backend_record['after_host_working_set_bytes'] / 2**20:,.2f}",
            ]
            thread_rows.append(
                "<tr><td>"
                + esc(backend_record["route"])
                + "</td>"
                + "".join("<td>" + cell_text + "</td>" for cell_text in cells)
                + "</tr>"
            )
        thread_changes = (
            """<h3>CPU worker-count comparison</h3>
<p>Automatic uses every logical processor available on the current machine. The addon also provides a saved manual limit from one worker to that maximum. These measurements used 32 workers; more workers do not guarantee a gain on other CPUs. Previous host preparation used 16 workers for both routes; it now uses the selected CPU count. SYCL's larger worker pool increases private commitment. Owned graph allocations are unchanged.</p>
<div class="table-wrap"><table><thead><tr><th>Route</th><th>Previous workers</th><th>Automatic workers</th><th>Previous frame ms</th><th>Automatic frame ms</th><th>Speedup</th><th>Previous private MiB</th><th>Automatic private MiB</th><th>Previous working set MiB</th><th>Automatic working set MiB</th></tr></thead><tbody>"""
            + "".join(thread_rows)
            + "</tbody></table></div>"
        )
    validation = data["validation"]
    noise = validation["original_numpy_noise"]
    worst = max(reference_record["max_abs"] for reference_record in noise)
    section = (
        """<!-- PORTABLE_CPP_BEGIN -->
<section id="portable-cpp" class="source">
<div class="eyebrow">Fresh compiled portable measurements · 2026-10-10</div>
<h1>CPU, CUDA, Vulkan, HIP and SYCL without Python</h1>
<p>The addon now executes these routes through compiled C++ libraries and an optional owned C++ helper. The actual inference processes loaded no Python or Torch. The optimized native RTX runner remains separate and unchanged.</p>
<div class="hint"><strong>Removing Python is not a speed optimization by itself.</strong> These C++ results include generic matrix implementations; read the selected engine and hardware for every row. Some routes are slower than the historical accelerated Python runners below. Memory accounting and image-equivalence limits are explicit.</div>
<p><strong>Image-equivalence limit:</strong> graph checks pass with identical noise, but C++ and NumPy generate slightly different noise. The separate original-noise test reached """
        + f"{worst:.6f}"
        + """ maximum absolute RGB proxy error. These runners are not certified as identical images to the old Python implementation.</p>
<p>1920 × 1080 input, 1920 × 1088 working grid · all eight transformer blocks · one pass · fixed synthetic RGB32F input, seed 0 and temporal history. Sequential isolated runner processes; warm samples exclude model loading, graph construction and the fractional-motion validation frame. No game FPS claim.</p>
<p><strong>Hardware:</strong> Ryzen 9 9950X3D, 16 cores / 32 threads, 96 GiB installed RAM; RTX 5080, driver 616.64; AMD Radeon integrated graphics (gfx1036), driver 32.0.21045.5002. SYCL CPU uses the Ryzen through a private Intel OpenCL CPU runtime. No Intel GPU was available.</p>
<div class="table-wrap"><table><thead><tr><th>C++ route and hardware</th><th>Operators ms</th><th>Complete C++ frame ms</th><th>Owned graph MiB</th><th>Host private MiB</th><th>Host working set MiB</th><th>Warm samples</th><th>Frame range ms</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></div>
<p class="metric-note">Operators use GPU events/timestamps, CPU wall intervals or SYCL CPU queue events as appropriate. CUDA/HIP/SYCL now measure contiguous stage intervals, including gaps between commands; previous per-command operator sums have a different scope. Complete frame boundaries are unchanged: preparation/history, transfers, graph work and resolve, excluding file I/O and initialization. SYCL CPU includes its broker round trip. Native RTX/NVIDIA use the separate captured-input GPU boundary described above; their portable complete-frame and graph fields are deliberately unavailable. Profiling remains enabled. Samples are limited; ranges are shown.</p>
"""
        + changes
        + thread_changes
        + """
<p><label>Stacked chart scale <select id="cpp-stack-scale"><option value="shares">Percentage shares</option><option value="absolute">Absolute time / memory</option></select></label></p>
<div class="chart-grid">
<section class="chart-panel"><h3>Speed comparison</h3><label>Metric <select id="cpp-time-metric"><option value="operator_ms">Operator intervals</option><option value="frame_ms">Complete C++ frame</option></select></label> <label>Scale <select id="cpp-time-scale"><option value="log">Logarithmic: log10(1 + ms)</option><option value="linear">Linear milliseconds</option></select></label><div id="cpp-speed-bars"></div><p class="metric-note">Fresh native RTX/NVIDIA references appear in operator view. Their different timing/input scopes are listed below.</p></section>
<section class="chart-panel"><h3>Memory comparison</h3><label>Metric <select id="cpp-memory-metric"><option value="graph_bytes">Owned graph allocations</option><option value="host_private_bytes">Host private commitment</option><option value="host_working_set_bytes">Host working set</option><option value="provider_transfer_buffer_bytes">Vulkan retained transfer buffer</option></select></label><div id="cpp-memory-bars"></div><p class="metric-note">All bars start at zero. Owned graph = unique packed-weight and scratch allocations, excluding driver/query pools, model-cache and host frame/history storage. NVIDIA Vulkan additionally retains a measured 63.75 MiB transfer buffer; shared-memory AMD uses none. This is separate from graph bytes, already reflected in process memory. CPU/SYCL allocations use system memory; integrated AMD allocations share system memory. Host metrics refer to the actual inference process after the final frame; native/NVIDIA reference rows use the mean of their separate captured-input runs. These scopes overlap and must not be added together. Missing metrics are omitted rather than treated as zero.</p></section>
<section class="chart-panel"><h3>Parts of a complete C++ frame</h3><div id="cpp-time-stacks"></div><div id="cpp-time-legend" class="legend"></div><p class="metric-note">100% bars: preparation/history, execution with transfers, resolve and other runtime work. Device operator time is contained within execution and is not added a second time.</p></section>
<section class="chart-panel"><h3>Parts of owned graph memory</h3><div id="cpp-memory-stacks"></div><div id="cpp-memory-legend" class="legend"></div><p class="metric-note">100% bars partition unique allocations. Scratch includes retained feature/skip/intermediate storage. Driver and process overhead remain unpartitioned.</p></section>
</div>
<div class="controls"><label>Inspect C++ route <select id="cpp-pie-backend"></select></label><label>Time breakdown <select id="cpp-pie-time"><option value="worker_parts">Complete C++ frame</option><option value="operator_parts">Measured network stages</option></select></label></div>
<div class="chart-grid"><section class="chart-panel"><h3>Time breakdown</h3><div id="cpp-time-pie"></div></section><section class="chart-panel"><h3>Owned graph memory breakdown</h3><div id="cpp-memory-pie"></div></section></div>
<details><summary>Validation, startup and comparison limits</summary>
<p>"""
        + str(validation["frames"])
        + """ full-output byte-exact regression checks across 96 × 64, odd 321 × 193 and 1920 × 1080 were recorded during this optimization pass. The final thread-control change reran thirty CPU/SYCL outputs; sixty unchanged GPU checks and their timings are retained from the same pass. The preserved baseline was independently checked against matched-noise Torch outputs, with maximum absolute error """
        + f"{validation['max_abs']:.9g}"
        + """; that independent comparison is inherited, not rerun here. Controls, block repeats, multiple passes, reset, temporal motion and validity masks were checked. All 658 decoded model tensors matched. All six addon offload routes passed physical D3D12 readback/output, motion accumulation, depth/reset/expiry and alpha checks with zero D3D12 debug errors. The addon mailbox and settings persistence passed in an owned ReShade host. Live-game visual validation is outside this test.</p>
<div class="hint"><strong>Matched-noise equivalence is not identical output to the original Python runner.</strong> C++ and NumPy Box-Muller math round some noise values differently. The separate original-NumPy-noise 1080p comparison reached """
        + f"{worst:.6f}"
        + """ maximum absolute RGB proxy error. This difference is not hidden by the matched-noise graph checks. See the JSON for all three original-noise comparisons.</div>
<div id="cpp-startup-table" class="table-wrap"></div>
<p>Initialization includes model resource validation/decoding and provider setup with the current filesystem cache. The first frame additionally builds/uploads the graph. Neither is cold-disk or game startup. The broker process has been audited directly for SYCL; its caller is measured separately in the raw local evidence. Cancellation, broker death, invalid controls/devices/model hashes and recovery passed compiled contract checks.</p>
<h3>Retained references</h3><div class="table-wrap"><table><thead><tr><th>Reference</th><th>GPU interval ms</th><th>Scope</th></tr></thead><tbody>"""
        + references
        + """</tbody></table></div>
<p>The fresh NVIDIA interval covers its entire Evaluate call; internal components remain unknown. Native RTX and NVIDIA share captured input, motion and controls in the first section, with three fresh processes each. Both include measured whole-process memory, but neither reference is a portable C++ complete-frame benchmark. The historical charts below retain their original dates and implementations.</p>
</details>
<p><a href="PORTABLE_CPP_RUNTIME.md">C++ build, configuration and validation scope</a> · <a href="PORTABLE_CPP_PERFORMANCE.json">New measurements, layers, hashes and alternatives</a></p>
</section>
<script id="portable-cpp-data" type="application/json">"""
        + json.dumps(data, separators=(",", ":")).replace("</", r"<\/")
        + """</script>
<style>
#portable-cpp h2{margin-top:12px}#portable-cpp h3{margin:8px 0 12px}#portable-cpp .chart-grid{margin:18px 0}#portable-cpp svg{width:100%;min-width:580px}#portable-cpp .chart-panel>div[id$="bars"],#portable-cpp .chart-panel>div[id$="stacks"]{overflow:auto}.cpp-pie{width:180px;height:180px;border-radius:50%;display:grid;place-items:center;margin:12px auto}.cpp-hole{width:105px;height:105px;background:var(--panel);border-radius:50%;display:grid;place-items:center;text-align:center;font-size:13px}.cpp-pie-row{display:grid;grid-template-columns:200px 1fr;align-items:center;gap:12px}.cpp-pie-row ul{padding-left:0;list-style:none}.cpp-pie-row li{margin:6px 0}@media(max-width:650px){.cpp-pie-row{grid-template-columns:1fr}}
</style>
<script>
(()=>{
const D=JSON.parse(document.getElementById('portable-cpp-data').textContent),byId=id=>document.getElementById(id),colors=['#79dbb4','#86b9ff','#ffc56e','#c9a3ff','#ed8ca3','#7ad6e0','#a6c972','#d9a56d'];
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),number=x=>x.toLocaleString(undefined,{maximumFractionDigits:2});
function bars(target,rows,key,unit,logarithmic=false){
 const valid=rows.filter(b=>Number.isFinite(b[key])).sort((a,b)=>a[key]-b[key]),maximum=Math.max(...valid.map(b=>b[key])),transform=v=>logarithmic?Math.log10(1+v):v,scale=v=>transform(v)/transform(maximum)*470;
 let s=`<svg viewBox="0 0 900 ${valid.length*43+44}" role="img" aria-label="${escape(key)} comparison; lower is better"><text x="270" y="16">${logarithmic?'log10(1 + ms), zero preserved':'Linear, zero baseline'} · ${unit}</text>`;
 valid.forEach((b,i)=>{let y=35+i*43;s+=`<text x="0" y="${y+17}">${escape(b.short_label||b.label)}</text><rect x="270" y="${y}" height="24" width="${Math.max(0.5,scale(b[key]))}" fill="${colors[i%colors.length]}"><title>${escape(b.label)}: ${number(b[key])} ${unit}; ${escape(b.hardware||b.scope)}</title></rect><text x="${280+scale(b[key])}" y="${y+17}">${number(b[key])}</text>`;});byId(target).innerHTML=s+'</svg>';
}
function stacks(target,legend,key,unit){
 let records=D.backends.map(b=>({...b,[key]:Object.fromEntries(Object.entries(b[key]).map(([k,v])=>[k,key==='memory_parts'?v/1048576:v]))})),names=[...new Set(records.flatMap(b=>Object.keys(b[key])))],maximum=Math.max(...records.map(b=>Object.values(b[key]).reduce((a,v)=>a+v,0))),absolute=byId('cpp-stack-scale').value==='absolute',s=`<svg viewBox="0 0 900 ${D.backends.length*43+20}" role="img" aria-label="${escape(key)} ${absolute?'absolute values':'percentage shares'}">`;
 records.forEach((b,i)=>{let y=10+i*43,total=Object.values(b[key]).reduce((a,v)=>a+v,0),x=270;s+=`<text x="0" y="${y+17}">${escape(b.short_label)}</text>`;names.forEach((name,j)=>{let value=b[key][name]||0,width=total?value/(absolute?maximum:total)*470:0;s+=`<rect x="${x}" y="${y}" width="${width}" height="24" fill="${colors[j%colors.length]}"><title>${escape(name)}: ${number(value)} ${unit}; ${number(total?100*value/total:0)}%</title></rect>`;x+=width;});s+=`<text x="750" y="${y+17}">${number(total)} ${unit}</text>`;});byId(target).innerHTML=s+'</svg>';byId(legend).innerHTML=names.map((n,i)=>`<span><i class="swatch" style="background:${colors[i%colors.length]}"></i>${escape(n)}</span>`).join('');
}
function pie(target,parts,unit){let total=Object.values(parts).reduce((a,v)=>a+v,0),position=0,segments=[],keys=[];Object.entries(parts).forEach(([name,value],i)=>{let end=position+360*value/total;segments.push(`${colors[i%colors.length]} ${position}deg ${end}deg`);position=end;keys.push(`<li><i class="swatch" style="background:${colors[i%colors.length]}"></i>${escape(name)}: ${number(value)} ${unit} (${number(100*value/total)}%)</li>`);});byId(target).innerHTML=`<div class="cpp-pie-row"><div class="cpp-pie" role="img" aria-label="${escape(keys.map(k=>k.replace(/<[^>]+>/g,'')).join('; '))}" style="background:conic-gradient(${segments.join(',')})"><div class="cpp-hole">${number(total)}<br>${unit}</div></div><ul>${keys.join('')}</ul></div>`;}
function draw(){let metric=byId('cpp-time-metric').value;bars('cpp-speed-bars',metric==='operator_ms'?[...D.backends,...D.references]:D.backends,metric,'ms',byId('cpp-time-scale').value==='log');let memory=byId('cpp-memory-metric').value;bars('cpp-memory-bars',[...D.backends,...D.references].filter(b=>Number.isFinite(b[memory])).map(b=>({...b,[memory]:b[memory]/1048576})),memory,'MiB');}
byId('cpp-pie-backend').innerHTML=D.backends.map(b=>`<option value="${escape(b.id)}">${escape(b.label+' · '+b.hardware)}</option>`).join('');
function pies(){let b=D.backends.find(b=>b.id===byId('cpp-pie-backend').value);pie('cpp-time-pie',b[byId('cpp-pie-time').value],'ms');pie('cpp-memory-pie',Object.fromEntries(Object.entries(b.memory_parts).map(([k,v])=>[k,v/1048576])),'MiB');}
['cpp-time-metric','cpp-time-scale','cpp-memory-metric'].forEach(id=>byId(id).addEventListener('change',draw));['cpp-pie-backend','cpp-pie-time'].forEach(id=>byId(id).addEventListener('change',pies));
function stacked(){stacks('cpp-time-stacks','cpp-time-legend','worker_parts','ms');stacks('cpp-memory-stacks','cpp-memory-legend','memory_parts','MiB');}byId('cpp-stack-scale').addEventListener('change',stacked);stacked();
byId('cpp-startup-table').innerHTML='<table><thead><tr><th>Route</th><th>Initialization ms</th><th>First frame ms</th><th>Clock</th></tr></thead><tbody>'+D.backends.map(b=>`<tr><td>${escape(b.label)}</td><td>${number(b.initialization_ms)}</td><td>${number(b.first_frame_ms)}</td><td>${escape(b.operator_clock)}</td></tr>`).join('')+'</tbody></table>';draw();pies();
})();
</script>
<!-- PORTABLE_CPP_END -->"""
    )
    document = args.html.read_text(encoding="utf-8")
    if "<!-- PORTABLE_CPP_BEGIN -->" in document:
        document = re.sub(
            r"<!-- PORTABLE_CPP_BEGIN -->.*?<!-- PORTABLE_CPP_END -->",
            lambda _: section,
            document,
            flags=re.S,
        )
    else:
        document = document.replace(
            '<div class="eyebrow">DLSS5 Phoenix · offline performance lab</div>',
            section
            + '\n<div class="eyebrow">Historical Python runner comparison and native RTX research</div>',
            1,
        )
    document = document.replace(
        "The main charts retain their historical measurements. The fresh source-built native comparison appears below;",
        "The fresh C++ portable comparison appears first. The older Python charts retain their historical measurements, and the source-built native RTX comparisons remain below;",
    )
    document = document.replace(
        "<h1>DLSS5 Phoenix performance, layer by layer.</h1>",
        "<h2>Historical layer-by-layer comparison</h2>",
    )
    args.html.write_text(document, encoding="utf-8")
    print(args.html)


if __name__ == "__main__":
    main()
