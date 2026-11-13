// ▶ Run tab: checks before running and how the loop runs (auto / queue).
import { h, fld, check, section, pill, hint, select, DEFAULTS, secondPassSize } from "./common.js";

function checksSection(c) {
  const { checks, tab } = c;
  const list = checks.value;
  const warns = list.filter(x => x.lvl === "warn").length;
  // one row per kind of issue, with the shots it concerns
  const groups = new Map();
  for (const x of list) { const k = `${x.lvl}|${x.text}`; if (!groups.has(k)) groups.set(k, { ...x, shots: [] }); groups.get(k).shots.push(x.shot); }
  const rows = [...groups.values()].sort((a, b) => (a.lvl === "warn" ? 0 : 1) - (b.lvl === "warn" ? 0 : 1));
  const go = i => { c.select(i); tab.value = "shots"; };
  return section("Checks", rows.length ? h("div", { class: "checks" }, rows.map(x => h("div", { class: "ck" }, [
    h("span", { class: "lvl" }, x.lvl === "warn" ? "⚠" : "ℹ"),
    h("span", { style: "flex:1;min-width:0" }, [h("div", x.text), h("div", { class: "row", style: "gap:3px;margin-top:3px" }, [
      ...x.shots.slice(0, 24).map(i => h("button", { class: "chip", title: "go to this shot", onClick: () => go(i) }, `#${i + 1}`)),
      x.shots.length > 24 ? hint(`+${x.shots.length - 24} more`) : null])]),
    pill(`${x.shots.length} shot(s)`, x.lvl === "warn" ? "warn" : ""),
  ]))) : h("div", { class: "hint" }, "✓ Nothing to fix in the shots that run."),
  { sub: "only shots that run", right: list.length ? pill(`${warns} warning(s) · ${list.length - warns} note(s)`, warns ? "warn" : "") : pill("all good", "ok") });
}

function loopSection(c) {
  const { plan, prog, active } = c;
  return section("Run", [
    h("div", { class: "grid" }, [
      fld("Run", select(plan.run, [["recursive", "One run, shot by shot (recommended)"], ["queue", "Queue loop (one shot per run)"],
        ["auto", "Auto loop (all shots at once)"]], v => c.setPlan("run", v, c.progress))),
      plan.run !== "auto" ? fld("Between shots", select(plan.between_shots || "keep", [["keep", "Keep models loaded (fastest)"],
        ["free", "Free memory: unload models (low VRAM)"]], v => c.setPlan("between_shots", v))) : null,
      fld("Skipped shots in the output", select(plan.skip_fill, [["original", "Keep original video"], ["drop", "Remove them"]], v => c.setPlan("skip_fill", v))),
    ]),
    h("div", { class: "row", style: "margin-top:6px" }, [
      check(plan.unload_before_join !== false, v => c.setPlan("unload_before_join", v), "Unload models before the join",
        "Once every shot is rendered, unload the models so the join (and the nodes after it) get the VRAM and RAM. Turn it off if a model runs again after BFS Shot Join (e.g. an upscale pass on the joined video): it would load again."),
    ]),
    plan.run !== "auto" ? h("div", { style: "margin-top:8px" }, [
      h("div", { class: "row", style: "margin-bottom:4px" }, [h("b", plan.run === "queue" ? "Queue loop" : "Shot by shot"), h("span", { class: "grow" }),
        hint(`${prog.done}/${prog.count || active.value.length} shots done`)]),
      h("div", { class: "prog" }, [h("div", { style: `width:${(100 * prog.done / Math.max(1, prog.count || active.value.length)).toFixed(1)}%` })]),
      h("div", { class: "row", style: "margin-top:6px" }, [
        plan.run === "queue" ? check(plan.auto_continue, v => c.setPlan("auto_continue", v), "Auto-queue the next shot") : null,
        h("button", { onClick: () => c.progress(false) }, "↻ Refresh status"), h("button", { title: "Cancel pending redo selections. Stored shots and loop progress are preserved.", onClick: () => c.progress(true) }, "Clear redo selection"),
      ]),
      hint(plan.run === "queue"
        ? "Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video."
        : "One Queue renders every shot in order inside the same run: each shot is stored, then BFS Shot Join starts the next one, so continuity can read the shot before. After a failure, Queue again to resume from the stored shots.",
        "display:block;margin-top:4px"),
    ]) : null,
  ], { sub: `${active.value.length} shot(s) will run` });
}

function memorySection(c) {
  const { memory, vram, size } = c;
  const m = memory.value;
  if (!m.level) return null;
  const k = n => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);
  const pct = vram.max_tokens ? Math.min(100, 100 * m.top.tok / (1.3 * vram.max_tokens)) : 0;
  return section("Memory", [
    h("div", { class: "grid" }, [
      fld("GPU", vram.gpu ? `${vram.gpu} · ${vram.total_gb} GB` : "—"),
      fld("Biggest shot in this plan", m.top.i >= 0 ? `#${m.top.i + 1} · ${m.top.gen} frames · ${size.w}×${size.h} · ${k(m.top.tok)} tokens` : "—"),
      fld("Biggest that ran here", vram.max_tokens ? `${k(vram.max_tokens)} tokens · peak ${vram.max_peak_gb} GB` : "nothing recorded yet"),
    ]),
    vram.max_tokens ? h("div", { class: "prog", style: "margin-top:6px", title: "100% = 1.3× the biggest shot that ran on this GPU" },
      [h("div", { style: `width:${pct.toFixed(1)}%;${m.level === "warn" ? "background:var(--warn,#d97706)" : ""}` })]) : null,
    hint({ ok: "✓ Every shot is within what already ran on this GPU.",
      note: "A bit bigger than anything that ran here: it will probably fit, maybe slower.",
      warn: "⚠ Much bigger than anything that ran here: it may run out of memory. Lower Megapixels or Max seconds / shot, use head 5, or set Between shots = free memory.",
      info: "No shot has run on this GPU yet: after the first one, every shot is compared with what already ran here." }[m.level],
      "display:block;margin-top:4px"),
  ], { open: m.level === "warn", sub: { ok: "within what ran here", note: "a bit bigger", warn: "may run out of memory", info: "no data yet" }[m.level] });
}

function cacheSection(c) {
  const { plan, cache, active } = c;
  if (plan.run === "auto") return null;
  const n = active.value.length, ok = cache.ok || [];
  const done = ok.filter(Boolean).length;
  const lossy = plan.shot_storage === "smaller" || plan.shot_storage === "smallest";
  return section("Stored shots", [
    h("div", { class: "grid" }, [
      fld("Shot cache", select(plan.shot_cache || "output", [["output", "Keep (output folder)"], ["temp", "Temp (lost when ComfyUI restarts)"]],
        v => c.setPlan("shot_cache", v, c.progress))),
      fld("Shot storage", select(plan.shot_storage || "lossless", [["lossless", "Lossless (FFV1, exact)"], ["smaller", "Smaller (H.264 RGB, visually lossless)"],
        ["smallest", "Smallest (H.264 RGB, light loss)"], ["raw", "Raw frames (.pt, biggest)"]], v => c.setPlan("shot_storage", v))),
    ]),
    lossy ? hint("⚠ The final video is assembled from the stored shots (and the continuity head reads them): a lossy storage "
      + "slightly changes the output. Lossless keeps it exact.", "display:block;margin-top:4px;color:var(--warn,#d97706)") : null,
    h("div", { class: "row", style: "margin-top:8px;flex-wrap:wrap;gap:4px" }, ok.length ? ok.map((g, k) => {
      const i = c.segs.value.indexOf(active.value[k]);
      return h("span", { class: ["pill", g ? "ok" : ""], title: g ? "stored and up to date" : "runs on the next Queue" }, [
        `#${(i >= 0 ? i : k) + 1} ${g ? "💾" : "⏳"}`,
        (cache.redo || []).includes(k) ? h("button", { class: "chip", title: "Cancel redo and keep the stored take",
          onClick: () => c.redo([k], true, true) }, "Undo redo") : g ? h("button", { class: "chip", style: "margin-left:4px", title: "generate this shot again on the next Queue"
          + (plan.redo_only ? " (only this shot)" : " (and the shots that continue from it)"),
          onClick: () => c.redo([k], !plan.redo_only) }, "↻") : null]);
    }) : [hint(n ? "Nothing stored yet for this video: Queue the workflow." : "")]),
    h("div", { class: "row", style: "margin-top:6px;flex-wrap:wrap;gap:6px" }, [
      hint(ok.length ? `${done}/${ok.length} stored and up to date · ${cache.mb || 0} MB` : ""),
      h("span", { class: "grow" }),
      check(!!plan.redo_only, v => c.setPlan("redo_only", v), "↻ redoes only that shot"),
      ok.length ? h("button", { title: "every stored shot runs again on the next Queue", onClick: () => c.redo(ok.map((_, k) => k), true) }, "↻ Redo all") : null,
      h("button", { disabled: !done, title: "Put the stored shots together into an MP4 in the output folder now, without generating: "
        + "one shot in memory at a time, the original video where a shot is skipped or not stored yet, with the soundtrack.",
        onClick: () => c.combine() }, "🎬 Combine stored shots"),
    ]),
    cache.combined ? h("div", { class: "row", style: "margin-top:6px" }, [
      hint(`✓ ${cache.combined.shots}/${cache.combined.of} shots → `),
      h("a", { href: cache.combined.url, target: "_blank", style: "margin-left:4px" }, cache.combined.filename)]) : null,
    hint("Each shot remembers the settings it was made with: the next Queue runs only the shots whose settings changed "
      + "(and the ones that continue from them). Changing what renders them (model, LoRA, seed, sampler, conditioning) runs every shot again; the join's own settings do not. "
      + (cache.folder ? `Folder: ${cache.folder}` : ""), "display:block;margin-top:6px"),
  ], { sub: ok.length ? `${done}/${ok.length} stored` : "nothing stored yet" });
}

// The planner only works out the second pass size; the graph upscales the latent (e.g. Minimax H3 Latent Upscaler
// (3D), target dimensions) and samples it again, reading the planner's second_pass / upscale_width / upscale_height
function secondPassSection(c) {
  const { plan, size } = c;
  const sp = { ...DEFAULTS.second_pass, ...(plan.second_pass || {}) };
  const set = (k, v) => c.setPlan("second_pass", { ...sp, [k]: v });
  const num = (k, step, min, max) => h("input", { type: "number", step, min, max, value: sp[k],
    onChange: e => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) set(k, v); } });
  const out = secondPassSize(size.w, size.h, sp);
  return section("Second pass (upscale)", [
    h("div", { class: "row" }, [check(sp.enabled, v => set("enabled", v), "Enable the second pass")]),
    h("div", { class: "grid", style: sp.enabled ? "" : "opacity:.55" }, [
      fld("Size", select(sp.mode, [["scale", "Scale by multiplier"], ["dims", "Target dimensions"], ["megapixels", "Megapixels"]], v => set("mode", v))),
      sp.mode === "scale" ? fld("Multiplier", num("scale", 0.05, 1, 4)) : null,
      sp.mode === "dims" ? fld("Width", num("width", 8, 64, 8192)) : null,
      sp.mode === "dims" ? fld("Height", num("height", 8, 64, 8192)) : null,
      sp.mode === "megapixels" ? fld("Megapixels", num("megapixels", 0.1, 0.1, 16)) : null,
      fld("Align (px)", num("align", 1, 1, 512), "32 recommended"),
      fld("Result", size.w ? `${size.w}×${size.h} → ${out.w}×${out.h}` : "—"),
    ]),
    hint("The planner outputs second_pass (on/off), upscale_width and upscale_height. Upscale the denoised latent of each shot "
      + "to that size (e.g. Minimax H3 Latent Upscaler (3D) in target dimensions mode) and sample it again before decoding; "
      + "use second_pass to switch that branch.", "display:block;margin-top:6px"),
  ], { open: !!sp.enabled, sub: sp.enabled ? `on · ${out.w}×${out.h}` : "off" });
}

export function runTab(c) {
  return h("div", { class: "card" }, [checksSection(c), memorySection(c), loopSection(c), secondPassSection(c), cacheSection(c)]);
}
