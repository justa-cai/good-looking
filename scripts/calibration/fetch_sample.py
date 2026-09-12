"""抓一批公有领域的人像照，用于标定几何指标的经验分布。

只用美国政府作品（Public domain）与 CC0 —— 这些可以自由使用。
下载到 tmp/faces/，脚本本身也放 tmp/，不进仓库。
"""
import json, re, sys, urllib.parse, urllib.request, pathlib

# Wikimedia 对没有 User-Agent 的请求直接 403
UA = {"User-Agent": "good-looking-research/0.1 (local calibration sample)"}
def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout)


CATS = [
    "Official_portraits_of_members_of_the_117th_United_States_Congress",
    "Official_portraits_of_members_of_the_116th_United_States_Congress",
]
OUT = pathlib.Path(__file__).parent / "faces" / "sample"
OUT.mkdir(parents=True, exist_ok=True)

api = "https://commons.wikimedia.org/w/api.php"
seen = {}
for cat in CATS:
    q = urllib.parse.urlencode({
        "action":"query","generator":"categorymembers",
        "gcmtitle":f"Category:{cat}","gcmtype":"file","gcmlimit":"60",
        "prop":"imageinfo","iiprop":"url|extmetadata","iiurlwidth":"1200","format":"json",
    })
    with get(f"{api}?{q}", timeout=40) as r:
        d = json.load(r)
    for v in (d.get("query") or {}).get("pages", {}).values():
        ii = (v.get("imageinfo") or [{}])[0]
        em = ii.get("extmetadata") or {}
        lic = em.get("LicenseShortName", {}).get("value", "")
        if not any(k in lic for k in ("Public domain", "CC0")):
            continue
        url = ii.get("thumburl")
        if not url:
            continue
        # 一个人只取一张：文件名去掉 (cropped)/(3x4) 之类的后缀当键
        title = v["title"]
        key = re.sub(r"\s*[\(\[]\s*(cropped|3x4|4x5|alt|alternate)[^\)\]]*[\)\]]", "", title, flags=re.I)
        key = re.sub(r"\.(jpg|jpeg|png)$", "", key, flags=re.I).strip().lower()
        # 优先选带 cropped 的（构图更适合做人脸测量），且跳过明显不是人像的
        if key not in seen or "cropped" in title.lower():
            seen[key] = (title, url)

print(f"候选 {len(seen)} 人", file=sys.stderr)
ok = 0
for i, (title, (t, url)) in enumerate(seen.items()):
    name = url.split("/")[-1].split("?")[0]
    name = urllib.parse.unquote(name)
    dest = OUT / f"{i:02d}_{re.sub(r'[^A-Za-z0-9._-]', '_', name)}"
    if dest.exists():
        ok += 1
        continue
    try:
        with get(url) as r:
            dest.write_bytes(r.read())
        ok += 1
    except Exception as e:
        print(f"失败 {name}: {e}", file=sys.stderr)
print(f"下载 {ok} 张到 {OUT}", file=sys.stderr)
