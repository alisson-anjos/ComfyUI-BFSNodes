// Shot Planner styles (one <style> tag, added once).
const CSS = `
.bsl{--bg:#16161a;--card:#1d1d23;--card2:#22222a;--line:#2d2d37;--line2:#3a3a46;--txt:#c9c9cf;--txt2:#8c8c97;--hi:#ececf1;
  --acc:#5b8cff;--acc2:#3a63e0;--ok:#7fe0a8;--warn:#ffc46b;--bad:#ff8f8f;
  font:12px/1.45 var(--font-family,system-ui,sans-serif);color:var(--txt);background:var(--bg);border-radius:10px;
  height:100%;overflow:auto;box-sizing:border-box;padding:10px;position:relative}
.bsl.root{position:absolute;inset:0;height:auto}
.bsl:focus{outline:none}
.bsl *{box-sizing:border-box}
.bsl .qm{color:#7d8bb0;cursor:help}
.bsl .grow{flex:1}
.bsl .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.bsl .col{display:flex;flex-direction:column;gap:6px}
.bsl .hint{font-size:10px;color:#7d7d88}
.bsl .chk{display:inline-flex;gap:5px;align-items:center;cursor:pointer;user-select:none}
.bsl .chk input{accent-color:var(--acc);margin:0}

/* header + step tabs */
.bsl .hdr{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.bsl .ttl{font-weight:600;color:var(--hi);font-size:13px;letter-spacing:.01em}
.bsl .tabs{display:flex;gap:4px;margin-bottom:8px;background:#111115;border:1px solid var(--line);border-radius:9px;padding:3px;position:sticky;top:-10px;z-index:7}
.bsl .tab{flex:1;background:none;border:0;border-radius:7px;padding:6px 6px;color:var(--txt2);font-size:11px;font-weight:600;display:flex;gap:5px;align-items:center;justify-content:center}
.bsl .tab:hover{background:#22222a;color:var(--txt)}
.bsl .tab.on{background:var(--acc2);color:#fff}
.bsl .tab .n{font-size:9px;font-weight:600;opacity:.75;border:1px solid currentColor;border-radius:999px;padding:0 5px;line-height:14px}
.bsl .tab .dot{width:6px;height:6px;border-radius:50%;background:var(--warn)}
.bsl .tab:disabled{opacity:.35}

/* pills, inputs, buttons */
.bsl .pill{font-size:10px;padding:1px 7px;border-radius:999px;background:#25252d;color:#a9a9b4;border:1px solid #33333d;white-space:nowrap}
.bsl .pill.ok{background:#173527;color:var(--ok);border-color:#245c40}
.bsl .pill.warn{background:#3a2a12;color:var(--warn);border-color:#6a4a16}
.bsl .pill.bad{background:#3a1a1d;color:var(--bad);border-color:#6a2c33}
.bsl .pill.acc{background:#1d2a4d;color:#a9c1ff;border-color:#2f4a8f}
.bsl select,.bsl input[type=number],.bsl input[type=text],.bsl textarea{background:#101014;border:1px solid #33333d;color:#e6e6ea;
  border-radius:6px;padding:4px 7px;font-size:11px;font-family:inherit;width:100%}
.bsl select:focus,.bsl input:focus,.bsl textarea:focus{border-color:var(--acc);outline:none}
.bsl textarea{resize:vertical;min-height:54px;font-family:ui-monospace,monospace}
.bsl input[type=range]{width:100%;accent-color:var(--acc)}
.bsl button{background:#26262e;border:1px solid #393945;color:#dcdce2;border-radius:6px;padding:4px 9px;font-size:11px;cursor:pointer;white-space:nowrap}
.bsl button:hover{background:#30303a}
.bsl button.pri{background:var(--acc2);border-color:#4b74f0;color:#fff}
.bsl button.pri:hover{background:#4672ee}
.bsl button.dng{background:#3a1f22;border-color:#6a2c33;color:#ffb3b3}
.bsl button.ghost{background:none;border-color:transparent;color:var(--txt2)}
.bsl button.ghost:hover{color:var(--txt);background:#26262e}
.bsl button:disabled{opacity:.45;cursor:default}
.bsl .seg-btns{display:inline-flex;border:1px solid #393945;border-radius:6px;overflow:hidden}
.bsl .seg-btns button{border:0;border-radius:0;border-right:1px solid #393945}
.bsl .seg-btns button:last-child{border-right:0}
.bsl .seg-btns button.on{background:var(--acc2);color:#fff}

/* cards and sections */
.bsl .card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:8px 10px;margin-bottom:8px}
.bsl .card h5{margin:0 0 6px;font-size:11px;font-weight:600;color:#b9b9c4;text-transform:uppercase;letter-spacing:.06em;display:flex;gap:6px;align-items:center}
.bsl .sec{border-top:1px solid var(--line);padding:8px 0 4px}
.bsl .sec:first-child{border-top:0;padding-top:2px}
.bsl .sech{display:flex;gap:6px;align-items:center;margin-bottom:6px;list-style:none;cursor:default}
.bsl summary.sech{cursor:pointer;user-select:none}
.bsl summary.sech::-webkit-details-marker{display:none}
.bsl .sect{font-size:11px;font-weight:600;color:#b9b9c4;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
.bsl .caret{display:inline-block;transition:transform .15s;color:var(--txt2);font-size:10px}
.bsl details[open]>summary .caret{transform:rotate(90deg)}
.bsl details:not([open])>.sech{margin-bottom:0}
.bsl .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px 8px}
.bsl .grid.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.bsl .fld label{display:block;font-size:10px;color:var(--txt2);margin-bottom:2px}
.bsl .err{color:var(--bad);background:#2a1517;border:1px solid #5a2228;border-radius:6px;padding:5px 8px;margin-bottom:8px;display:flex;gap:8px;align-items:center}
.bsl .note{background:#1a2034;border:1px solid #2b3557;border-radius:7px;padding:6px 8px;color:#b8c4e6;font-size:11px}
.bsl .note b{color:#dfe6ff}
.bsl .empty{padding:26px 10px;text-align:center;color:var(--txt2)}
.bsl .empty b{display:block;color:var(--hi);font-size:13px;margin-bottom:4px}

/* busy bar */
.bsl .busybar{position:sticky;top:0;z-index:8;margin:6px 0;padding:6px 8px;border-radius:8px;background:#22222b;border:1px solid #3d3d4a}
.bsl .busybar .brow{display:flex;gap:6px;align-items:center}
.bsl .busybar .spin{width:12px;height:12px;border-radius:50%;border:2px solid var(--acc);border-top-color:transparent;animation:bslspin .8s linear infinite;flex:none}
.bsl .busybar .bprog{height:4px;background:#33333d;border-radius:3px;margin-top:5px;overflow:hidden;position:relative}
.bsl .busybar .bprog>div{height:100%;background:var(--acc);transition:width .2s}
.bsl .busybar .bprog.ind>div{position:absolute;width:30%;animation:bslind 1.2s ease-in-out infinite}
@keyframes bslspin{to{transform:rotate(360deg)}}
@keyframes bslind{0%{left:-30%}100%{left:100%}}

/* timeline */
.bsl .tl{position:relative;overflow-x:auto;overflow-y:hidden;border-radius:7px;background:#111115;border:1px solid #2a2a33}
.bsl .tlin{position:relative}
.bsl .strip{display:flex;height:64px;overflow:hidden}
.bsl .strip img{height:64px;object-fit:cover;flex:none;opacity:.92}
.bsl .spark{display:block;height:26px;width:100%}
.bsl .segs{position:relative;height:30px;margin-top:2px;cursor:crosshair}
.bsl .seg{position:absolute;top:2px;bottom:2px;border-radius:5px;display:flex;align-items:center;justify-content:center;
  font-size:10px;color:#fff;font-weight:600;overflow:hidden;cursor:pointer;border:2px solid transparent;user-select:none}
.bsl .seg.sel{border-color:#fff;box-shadow:0 0 0 2px #5b8cff66}
.bsl .seg.off{opacity:.35;background-image:repeating-linear-gradient(45deg,#0004 0 6px,#0000 6px 12px)}
.bsl .seg.long{outline:2px solid #ff5d5d;outline-offset:-2px}
.bsl .seg.playing{box-shadow:0 0 0 2px #ffd34d}
.bsl .hdl{position:absolute;top:-70px;bottom:0;width:9px;margin-left:-4px;cursor:ew-resize;z-index:3}
.bsl .hdl::after{content:"";position:absolute;left:3px;top:0;bottom:0;width:3px;background:#fff;opacity:.85;border-radius:2px;box-shadow:0 0 4px #000}
.bsl .hdl:hover::after{background:#ffd34d}
.bsl .rgrid{display:grid;grid-template-columns:auto 130px auto 130px auto;gap:6px 10px;align-items:start;justify-content:start}
.bsl .rgrid>button{margin-top:15px}
.bsl .rgrid .fld .hint{margin-top:2px}
.bsl .outr{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(45deg,#000a 0 6px,#0007 6px 12px);z-index:2;pointer-events:none}
.bsl .cut{position:absolute;top:0;height:64px;width:2px;background:#ff4d6d;opacity:.9;pointer-events:none}
.bsl .ph{position:absolute;top:0;bottom:0;width:1px;background:#ffd34d;pointer-events:none;z-index:4}
.bsl .playhead{position:absolute;top:0;bottom:0;width:2px;background:#ffd34d;box-shadow:0 0 6px #ffd34d;pointer-events:none;z-index:5}
.bsl .tip{position:absolute;z-index:6;pointer-events:none;background:#0d0d10ee;border:1px solid #3a3a44;border-radius:6px;padding:3px;font-size:10px;color:#ddd}
.bsl .tip img{display:block;height:84px;border-radius:4px}

/* shot cards */
.bsl .shots{display:flex;gap:6px;overflow-x:auto;padding:2px 2px 6px;scroll-snap-type:x proximity}
.bsl .shots .sc{flex:0 0 158px;scroll-snap-align:start}
.bsl .sc{background:var(--card2);border:1px solid #30303a;border-radius:8px;padding:6px;cursor:pointer;position:relative}
.bsl .sc:hover{border-color:#4a4a58}
.bsl .sc.sel{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}
.bsl .sc.skip{opacity:.62}
.bsl .sc .bar{height:3px;border-radius:2px;margin-bottom:5px}
.bsl .sc .t{font-size:10px;color:#9a9aa6}
.bsl .sc .p{font-size:10px;color:#c9c9d3;margin-top:3px;height:28px;overflow:hidden}
.bsl .sc .play{position:absolute;top:5px;right:5px;padding:1px 6px;font-size:10px}
.bsl .badges{display:flex;gap:3px;flex-wrap:wrap;margin-top:4px}
.bsl .bdg{font-size:9px;padding:0 5px;border-radius:4px;background:#2a2a33;color:#a9a9b4;line-height:15px}
.bsl .bdg.on{background:#1d2a4d;color:#a9c1ff}
.bsl .bdg.warn{background:#3a2a12;color:var(--warn)}
.bsl .thumbs{display:flex;gap:4px;margin-top:4px}
.bsl .rt{width:34px;height:34px;border-radius:5px;object-fit:cover;background:#2a2a33;border:1px solid #3a3a45}
.bsl .rt.ph2{display:flex;align-items:center;justify-content:center;color:#666;font-size:9px}
.bsl .who{display:flex;gap:3px;margin-top:3px;align-items:center}
.bsl .who img{width:22px;height:22px;border-radius:50%;object-fit:cover;border:1px solid #3a3a45;opacity:.7}
.bsl .who img.main{width:26px;height:26px;opacity:1;border-color:#fff}
.bsl .who img.lk{border-color:#8fd18f}

/* shot editor */
.bsl .edhd{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.bsl .edhd .big{font-size:14px;font-weight:700;color:var(--hi)}
.bsl .refbox{display:flex;gap:8px;align-items:center}
.bsl .refslot{width:78px;height:78px;border-radius:8px;border:1px dashed #44444f;background:#141418;display:flex;align-items:center;
  justify-content:center;cursor:pointer;overflow:hidden;color:#6f6f7a;font-size:10px;text-align:center;flex:none;white-space:pre-line}
.bsl .refslot img{width:100%;height:100%;object-fit:cover}
.bsl .refslot:hover{border-color:var(--acc)}
.bsl .refslot.sm{width:56px;height:56px;font-size:10px}
.bsl .refslot.via{border-style:solid;border-color:#3b5b3b}
.bsl .refslot{position:relative}
.bsl .refslot .rnum{position:absolute;left:3px;top:2px;font-size:10px;padding:0 4px;border-radius:4px;background:#000a;color:#fff}
.bsl .refslot.add{font-size:22px;color:#8a8a99}
.bsl .refbox{flex-wrap:wrap;max-width:100%;align-items:flex-start}
.bsl .pchips{display:flex;gap:4px;flex-wrap:wrap;margin-top:4px}
.bsl .pchip{display:inline-flex;gap:1px}
.bsl .pchip button{display:inline-flex;align-items:center;gap:3px;padding:1px 5px;font-size:10px}
.bsl .pchip img{width:16px;height:16px;border-radius:3px;object-fit:cover}
.bsl .rslot{display:flex;flex-direction:column;gap:3px;align-items:center}
.bsl .guidecolor{display:flex;flex-direction:column;align-items:center;gap:2px;max-width:112px}
.bsl .guidecolor select{width:100%;font-size:10px;padding:2px 4px}
.bsl .rbtns{display:flex;gap:3px;min-height:20px}
.bsl .rbtns button{padding:1px 6px;font-size:10px}
.bsl .recent{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;align-items:center}
.bsl .recent img{width:30px;height:30px;border-radius:5px;object-fit:cover;cursor:pointer;border:1px solid #3a3a45}
.bsl .recent img:hover{border-color:var(--acc)}
.bsl .recent.sm img{width:22px;height:22px}
.bsl .modes{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;margin-bottom:6px}
.bsl .mode{border:1px solid var(--line2);border-radius:7px;padding:6px 8px;cursor:pointer;background:#18181d}
.bsl .mode:hover{border-color:#55556a}
.bsl .mode.on{border-color:var(--acc);background:#1a2034}
.bsl .mode b{display:block;color:var(--hi);font-size:11px}
.bsl .mode span{font-size:10px;color:var(--txt2)}
.bsl .mstrip{display:flex;gap:4px;margin-top:6px;align-items:center;flex-wrap:wrap}
.bsl .mstrip img{height:90px;border-radius:5px;border:1px solid #3a3a45}
.bsl .vsug{margin-top:6px;padding:6px 8px;border:1px dashed #4a4a58;border-radius:6px;display:flex;flex-direction:column;gap:3px}
.bsl .vsug button{margin-left:6px;padding:1px 6px}
.bsl .copy{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.bsl .player{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}
.bsl .player video{max-height:220px;max-width:100%;border-radius:8px;background:#000;border:1px solid var(--line)}
.bsl .pinfo{flex:1;min-width:200px;display:flex;flex-direction:column;gap:6px}
.bsl .tc{font-family:ui-monospace,monospace;font-size:12px;color:#e6e6ea;background:#101014;border:1px solid var(--line);border-radius:6px;padding:6px 8px;line-height:1.6}
.bsl .tc b{color:#ffd34d}

/* cast, VLM, checks */
.bsl .cast{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.bsl .person{background:#141418;border:1px solid var(--line);border-radius:8px;padding:6px;display:flex;flex-direction:column;gap:4px;align-items:flex-start}
.bsl .person.ign{opacity:.45}
.bsl .person .face{width:64px;height:64px;border-radius:50%;object-fit:cover;cursor:pointer;border:2px solid #3a3a45}
.bsl .person .t{font-size:10px;color:#9a9aa6}
.bsl .dlist{display:flex;flex-direction:column;gap:6px;margin-top:6px}
.bsl .drow{display:flex;gap:8px;align-items:flex-start}
.bsl .dthumbs{display:flex;gap:3px}
.bsl .dthumbs img{width:44px;height:58px;object-fit:cover;border-radius:5px;border:1px solid #3a3a45}
.bsl .checks{display:flex;flex-direction:column;gap:3px}
.bsl .chip{padding:0 6px;font-size:10px;line-height:16px;border-radius:4px}
.bsl .ck{display:flex;gap:8px;align-items:flex-start;padding:4px 6px;border-radius:6px;background:#18181d;border:1px solid var(--line)}
.bsl .ck .lvl{font-size:12px}
.bsl .prog{height:7px;border-radius:4px;background:#26262e;overflow:hidden}
.bsl .prog>div{height:100%;background:linear-gradient(90deg,#3a63e0,#7fe0a8)}
.bsl .langpick{display:flex;gap:5px;align-items:center}
.bsl .langpick select{width:auto;min-width:100px;padding:4px 6px}
.bsl .tourback{position:fixed;inset:0;z-index:10001;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:16px}
.bsl .tourbox{width:min(480px,94vw);background:#1d1d23;border:1px solid #4a4a58;border-radius:12px;padding:16px;box-shadow:0 12px 48px #0009}
.bsl .tourbox h3{margin:18px 0 8px;color:var(--hi)}
.bsl .tourbox p{line-height:1.55}

/* points window (teleported to <body>) */
.bsl .mmodal{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center}
.bsl .mbox{background:#1d1d23;border:1px solid #3a3a45;border-radius:10px;padding:12px;width:min(960px,94vw);height:min(900px,94vh);overflow:auto;box-sizing:border-box}
/* the frame fits a fixed stage whatever the video's size or orientation; points are stored relative to the frame */
.bsl .mstage{display:flex;justify-content:center;align-items:center;height:min(540px,56vh);background:#0d0d10;border-radius:6px}
.bsl .mimg{position:relative;cursor:crosshair;user-select:none;line-height:0;display:inline-block}
.bsl .mimg img{display:block;max-width:100%;max-height:min(540px,56vh);border-radius:6px}
.bsl .mthumb{width:22px;height:22px;border-radius:4px;object-fit:cover;vertical-align:middle;border:2px solid var(--mark,#555)}
.bsl .pt{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;border:2px solid #fff;cursor:pointer}
.bsl .pt.pos{background:#3ccf6b}.bsl .pt.neg{background:#e8455a}
.bsl .pt.cg.pos{background:var(--point-color,#00ff00)}
.bsl .pt.other{opacity:.5}
.bsl .kslider{position:relative;flex:1;min-width:160px;padding-bottom:8px}
.bsl .kmarks{position:absolute;left:8px;right:8px;bottom:0;height:8px}
.bsl .kmark{position:absolute;width:8px;height:8px;margin-left:-4px;border-radius:50%;background:var(--mark,#3ccf6b);opacity:.55;cursor:pointer;border:1px solid #000}
.bsl .kmark.on{opacity:1}
.bsl .kchip{display:inline-flex;gap:1px}.bsl .kchip.on button:first-child{border-color:var(--acc)}
.bsl .mreg{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:6px}
`;

export function styles() {
  if (document.getElementById("bfs-shotloop-css")) return;
  const el = document.createElement("style");
  el.id = "bfs-shotloop-css";
  el.textContent = CSS;
  document.head.appendChild(el);
}
