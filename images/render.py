import json, math, subprocess, sys

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAD, SCALE = 24, 2
DARK_BG = "#121212"

HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>body{margin:0;padding:0;background:%s;}</style></head><body>
<script type="module">
try {
  const Lib = await import("https://esm.sh/@excalidraw/excalidraw@0.18.0?bundle");
  window.EXCALIDRAW_ASSET_PATH = "https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/prod/";
  const scene = __SCENE__;
  const svg = await Lib.exportToSvg({
    elements: scene.elements,
    appState: Object.assign({}, scene.appState||{}, {exportBackground:true, exportWithDarkMode:__DARK__, viewBackgroundColor:"__XBG__"}),
    files: scene.files||null,
    exportPadding: __PAD__,
  });
  const w=parseFloat(svg.getAttribute("width")), h=parseFloat(svg.getAttribute("height"));
  svg.setAttribute("width", w*__SCALE__); svg.setAttribute("height", h*__SCALE__);
  document.body.appendChild(svg);
  await document.fonts.ready;
} catch(e){ document.body.textContent="ERR "+(e&&e.message+" "+e.stack); }
</script></body></html>"""

def bounds(els):
    xs,ys,xe,ye=[],[],[],[]
    for e in els:
        if e.get("isDeleted"): continue
        xs.append(e["x"]); ys.append(e["y"])
        xe.append(e["x"]+e.get("width",0)); ye.append(e["y"]+e.get("height",0))
    return min(xs),min(ys),max(xe),max(ye)

def render(src, png, dark=True):
    scene=json.load(open(src))
    x0,y0,x1,y1=bounds(scene["elements"])
    w=math.ceil((x1-x0+2*PAD)*SCALE); h=math.ceil((y1-y0+2*PAD)*SCALE)
    bg=DARK_BG if dark else "#ffffff"
    html=(HTML % bg).replace("__SCENE__",json.dumps(scene)).replace("__PAD__",str(PAD)).replace("__SCALE__",str(SCALE)).replace("__DARK__","true" if dark else "false").replace("__XBG__", "#ffffff" if dark else "#ffffff")
    open("/tmp/exca/page.html","w").write(html)
    r=subprocess.run([CHROME,"--headless=new","--disable-gpu",f"--screenshot={png}",
        f"--window-size={w},{h}",f"--default-background-color={'121212FF' if dark else 'FFFFFFFF'}",
        "--hide-scrollbars","--virtual-time-budget=60000","--timeout=120000","file:///tmp/exca/page.html"],
        capture_output=True,text=True,timeout=180)
    if r.returncode!=0: sys.exit("chrome failed: "+r.stderr[-400:])
    print(f"wrote {png} ({w}x{h})")

if __name__=="__main__":
    render(sys.argv[1], sys.argv[2])
