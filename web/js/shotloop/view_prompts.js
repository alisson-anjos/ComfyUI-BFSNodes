// 📝 Pictures & prompt tab: the pictures and prompt every shot uses, 🤖 Auto prompt, and the VLM helpers (collapsed).
import { h, fld, check, section, pill, hint, select, refBox, promptChips, MARK_PALETTE, pictureColor } from "./common.js";

const PRESETS = ["short", "object", "full body", "head / face", "face attributes", "outfit", "scene / setting", "custom"];

function globalSection(c) {
  const { plan } = c;
  // Colored Mark: every picture has its mark color (a person marked in it becomes that picture)
  const marker = k => h("label", { class: "guidecolor" }, [
    h("span", { class: "hint" }, "Mark color"),
    h("select", { value: pictureColor(plan, k + 1), title: `Colored Mark color of <Picture ${k + 1}>`,
      onChange: e => c.setPictureColor(k + 1, e.target.value) },
      MARK_PALETTE.map(([v, name]) => h("option", { value: v }, name))),
  ]);
  return section("Pictures & prompt", [
    refBox(c, "global", plan.global_refs, [], marker),
    h("div", { class: "pbox", style: "margin-top:6px" }, [
      h("textarea", { style: "min-height:78px", placeholder: plan.auto_prompt?.enabled
        ? "Request for every shot (🤖 Auto prompt writes the H3 prompt from it), e.g. 'replace the bottle with the red can from Picture 2'"
        : "Global prompt (a connected `prompt` input overrides it)",
        value: plan.global_prompt, onChange: e => c.setPlan("global_prompt", e.target.value) }),
      promptChips(c, plan.global_refs.filter(Boolean), v => c.setPlan("global_prompt", v)),
    ]),
    h("div", { class: "hint", style: "margin-top:4px" }, [
      "Click + or drop images. Placeholders: ", h("b", "{target}"), " what changes (per shot) · ", h("b", "{details}"), " / ",
      h("b", "{details@Picture n}"), " picture descriptions · ", h("b", "{shot}"), " the VLM's shot description · ",
      h("b", "{setting}"), " the setting picture. A connected prompt input overrides this prompt."]),
  ], { sub: "every shot uses these unless it has its own" });
}

function vlmSection(c) {
  const { plan, vlmSug, busy, pictures } = c;
  const VC = plan.vlm_cfg || {};
  const opt = (k, label, title) => check(VC[k], v => c.setVlmCfg(k, v), label, title);
  const sugN = Object.keys(vlmSug.value).length;
  return [
    section("VLM", [
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        opt("enabled", "Use the VLM when the workflow runs", "At run time the planner asks the VLM about the shots it runs and applies the options below"),
        opt("auto_segment", "Fill the selection text", "Shots without a selection text or points get the VLM's suggestion"),
        opt("auto_shot", "Fill {shot} in the prompt", "Write {shot} in a prompt: it becomes the VLM's description of that shot (camera, framing, action)"),
      ]),
      h("div", { class: "grid" }, [
        fld("Frames per shot", h("input", { type: "number", min: 1, max: 8, step: 1, value: VC.frames, onChange: e => c.setVlmCfg("frames", parseInt(e.target.value) || 3) })),
        fld("Max tokens", h("input", { type: "number", min: 64, max: 2048, step: 32, value: VC.max_tokens, onChange: e => c.setVlmCfg("max_tokens", parseInt(e.target.value) || 1024) })),
        fld("Image max side (VLM)", select(String(VC.image_max_side || 1024),
          [448, 672, 896, 1024, 1280, 1536, 2048].map(n => [String(n), `${n} px`]),
          v => c.setVlmCfg("image_max_side", parseInt(v) || 1024)), "VLM image sizing"),
      ]),
      h("textarea", { style: "margin-top:6px;min-height:44px", placeholder: "Extra instruction for the VLM (optional), e.g. 'segment the woman, not the man'",
        value: VC.instruction || "", onChange: e => c.setVlmCfg("instruction", e.target.value) }),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("button", { class: "pri", disabled: !!busy.value, onClick: c.analyseVLM }, "🤖 Analyse shots"),
        h("button", { disabled: !sugN, onClick: c.applyAllSug, title: "The suggested selection text, on every shot without one" }, "Apply suggestions → all"),
      ]),
      hint("Connect a Qwen3-VL to the planner's vlm input and run the workflow once: then these buttons work.", "display:block;margin-top:4px"),
    ], { open: false, sub: "suggests each shot's selection and a description",
         right: h("span", { class: "row" }, [VC.enabled ? pill("on", "ok") : pill("off"), sugN ? pill(`${sugN} suggestions`) : null]) }),
    section("Describe references ({details})", [
      h("div", { class: "row", style: "gap:8px" }, [
        fld("Describe", select(VC.describe_preset, PRESETS, v => c.setVlmCfg("describe_preset", v), { style: "width:auto" }),
          "what every picture gets, unless it has its own below"),
        h("button", { class: "pri", disabled: !!busy.value || !pictures.value.length, onClick: () => c.describePictures() }, "📝 Describe refs"),
        h("button", { disabled: !!busy.value || !pictures.value.length,
          title: "Try other descriptions: a new seed (and some temperature if it is 0), then describe again",
          onClick: () => { if (!(+VC.temperature > 0)) c.setVlmCfg("temperature", 0.7); c.setVlmCfg("seed", (+VC.seed || 0) + 1); c.describePictures(); } }, "🎲"),
        fld("Temperature", h("input", { type: "number", min: 0, max: 1.5, step: 0.1, value: VC.temperature ?? 0, style: "width:70px",
          title: "0 = always the same answer (greedy); 0.5-1 = varied answers, picked by the seed",
          onChange: e => c.setVlmCfg("temperature", Math.max(0, parseFloat(e.target.value) || 0)) })),
        fld("Seed", h("input", { type: "number", min: 0, step: 1, value: VC.seed ?? 0, style: "width:80px",
          title: "Used when the temperature is above 0: another seed, another description",
          onChange: e => c.setVlmCfg("seed", Math.max(0, parseInt(e.target.value) || 0)) })),
        hint(`${pictures.value.length} picture(s) · editable below`),
        check(VC.fit_prompt !== false, v => c.setVlmCfg("fit_prompt", v), "fit the prompt",
          "The VLM gets the global prompt's sentence about that picture ({details}, {details2} or {details@Picture n}; and the shot prompt's {target} sentence for 🧑 Describe target) with a blank in its place, and writes the description to fit it: the description adapts to the prompt, not the other way round."),
      ]),
      VC.describe_preset === "custom" ? h("textarea", { style: "margin-top:6px;min-height:52px", value: VC.describe_custom || "",
        placeholder: "Your instruction for the VLM, e.g. 'Describe the character's costume and props piece by piece…'",
        onChange: e => c.setVlmCfg("describe_custom", e.target.value) }) : null,
      pictures.value.length ? h("div", { class: "dlist" }, pictures.value.map(x => {
        const own = plan.picture_cfg?.[x.name] || {};
        return h("div", { class: "drow" }, [
          h("div", { class: "dthumbs" }, [h("img", { src: c.viewUrl(x.name) })]),
          h("div", { style: "flex:1" }, [
            h("div", { class: "row", style: "gap:6px;margin-bottom:3px" }, [
              hint(`Picture n in ${x.where.length > 4 ? `${x.where.slice(0, 4).join(", ")} +${x.where.length - 4}` : x.where.join(", ")}`),
              h("span", { class: "grow" }),
              select(own.preset || "", [["", `default (${VC.describe_preset || "short"})`], ...PRESETS],
                v => c.setPictureCfg(x.name, "preset", v), { style: "width:auto", title: "What to describe in this picture" }),
              h("button", { class: "sm", disabled: !!busy.value, title: "Describe only this picture again",
                onClick: () => c.describePictures([x.name]) }, "🔄"),
            ]),
            own.preset === "custom" ? h("textarea", { style: "min-height:36px;margin-bottom:3px", value: own.custom || "",
              placeholder: "Instruction for this picture, e.g. 'Describe only the outfit, piece by piece'",
              onChange: e => c.setPictureCfg(x.name, "custom", e.target.value) }) : null,
            h("textarea", { style: "min-height:44px", value: plan.picture_details?.[x.name] || "",
              placeholder: "no description yet: 📝 Describe refs, 🔄, write your own, or leave it to the VLM at run time",
              onChange: e => c.setPictureDetail(x.name, e.target.value) }),
          ]),
        ]);
      })) : hint("Add a reference (global or per shot) first."),
    ], { open: false, sub: "the text of {details} / {details@Picture n}" }),
  ];
}

// 🤖 Auto prompt: the VLM writes each shot's prompt in H3's full-reference format from key frames of the shot, its
// pictures and their descriptions, the request in the prompt box and the shot's marks and mode
function autoSection(c) {
  const { plan, busy, segs } = c;
  const A = plan.auto_prompt || {};
  const S = segs.value, missing = S.filter(s => !c.writtenOf(s)?.text).length;
  const stale = S.filter(s => c.writtenOf(s)?.text && c.writtenOf(s).for !== c.autoFor(s)).length;
  return section("🤖 Auto prompt (H3)", [
    h("div", { class: "row", style: "gap:12px;align-items:center" }, [
      check(A.enabled, v => c.setAuto("enabled", v), "Write the prompts with the VLM",
        "On: the prompt boxes (global and per shot) hold your request ('change the bottle into a red can', 'Picture 1 dances in the rain'); the VLM writes each shot's MiniMax H3 prompt from key frames of the shot, the pictures and their descriptions, the request and the shot's Colored Marks, selection and mode, in H3's official full-reference format."),
      fld("Style", select(A.style || "h3", [["h3", "H3 official (six sections)"], ["captions", "Short edit request"]],
        v => c.setAuto("style", v), { style: "width:auto" }), "Choose prompt format. The guide performance stays fixed; request style, setting, lighting, outfit or camera changes explicitly. Identify setting pictures in the request; bind colors only to character references."),
      fld("Trigger", select(A.trigger ?? "motion_transfer", [["motion_transfer", "motion_transfer:"], ["body_swap", "body_swap:"], ["", "None (plain prompt)"]],
        v => c.setAuto("trigger", v), { style: "width:auto" }), "Match the LoRA caption prefix. Choose None for a model trained without a trigger."),
      fld("Key frames", select(String(A.frames || 0), [["0", "auto (by length)"], ...[3, 4, 5, 6, 8, 10, 12].map(n => [String(n), String(n)])],
        v => c.setAuto("frames", parseInt(v) || 0), { style: "width:auto" }), "auto: about 1.5 per second, 3 to 8"),
      fld("Source video in the prompt", select(A.source || "guide", [["guide", "guide video (aligned guide)"], ["video", "<Video 1> (native reference video)"]],
        v => c.setAuto("source", v), { style: "width:auto" }), "match the conditioning node's guide_mode"),
      h("button", { class: "pri", disabled: !!busy.value || !S.length, onClick: () => c.writePrompts() }, "🤖 Write all prompts"),
      h("button", { disabled: !!busy.value || !missing, onClick: () => c.writePrompts(S.map((s, i) => (c.writtenOf(s)?.text ? -1 : i)).filter(i => i >= 0)) },
        `only the missing (${missing})`),
    ]),
    A.enabled && A.template ? h("div", { class: "hint", style: "display:block;margin-top:4px", title: A.template },
      `Template (the node's prompt input, last run): ${A.template.slice(0, 110)}${A.template.length > 110 ? "…" : ""} · the prompt boxes are the request applied to it`) : null,
    hint(A.enabled ? `${S.length - missing}/${S.length} shots have a written prompt${stale ? ` · ${stale} older than their request, pictures or marks` : ""} · shots without one are written when the workflow runs (with the VLM on the planner) · edit them in each shot (Shots tab)`
      : "Off: the prompt boxes are the prompts.", "display:block;margin-top:4px"),
  ], { sub: "request + video + pictures -> MiniMax H3 prompt", right: A.enabled ? pill("on", "ok") : pill("off") });
}

export function promptsTab(c) {
  return h("div", { class: "card" }, [globalSection(c), autoSection(c), ...vlmSection(c)]);
}
