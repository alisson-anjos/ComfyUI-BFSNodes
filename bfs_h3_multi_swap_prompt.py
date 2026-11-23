"""BFS H3 Multi Swap Prompt (experimental): the prompt of the multi-character swap LoRA (v7), where each person to
replace is marked with colored hatching in only a few moments of the guide video (Video Segment Picker, settings mode
'sparse hatching'). Subject j is marked in color j and takes its appearance from Picture j."""
import json

COLOR_NAMES = ["red", "blue", "green", "yellow", "magenta", "cyan"]
MOMENTS = ("The marks appear only in a few moments of the guide video; between them each subject is the same person, "
           "followed through the whole video wherever it moves.")


def multi_swap_prompt(subjects):
    """subjects: subject numbers (1..6, color j = Picture j) -> the training prompt."""
    subjects = sorted({int(s) for s in subjects if 1 <= int(s) <= len(COLOR_NAMES)}) or [1]
    defs = " ".join(f"<Subject {j}> is the person marked by the {COLOR_NAMES[j - 1]} hatching lines and matching "
                    f"{COLOR_NAMES[j - 1]} contour in the guide video; its complete appearance comes from <Picture {j}>."
                    for j in subjects)
    k = len(subjects)
    who = "the marked person" if k == 1 else f"each of the {k} marked people"
    return ("character_swap: subject_definitions: The guide video is the source video. Colored hatching lines and matching "
            "contours identify the subjects to replace; the visible pixels between the lines provide pose, motion, expression, "
            f"lighting and scene context. {MOMENTS} {defs} summary: [video editing + reference generation] Replace {who} with "
            "their assigned reference character from the first frame to the last. Remove all colored lines and contours from "
            "the result. retention_analysis: Transfer the face, hair, skin, body proportions and complete outfit from each "
            "assigned reference. Preserve the source pose, motion, hand positions, head orientation, gaze and facial expression. "
            "Adapt each replacement to the source lighting, shadows and exposure. Do not copy the pose or lighting of the "
            "reference. Do not retain the original marked subject's identity or clothing. Never mix features between subjects. "
            "Preserve every unmarked person, the background, held objects, foreground occluders and the camera motion. The "
            "hatching identifies who changes and does not constrain the replacement silhouette.")


def face_swap_prompt_v2(names, image=False):
    """Caption v2 of the multi face swap LoRAs (datasets with the colors at random cell positions): the color is stated as
    the only link between a face and its reference, every replacement repeats its color and the hair is kept. Must stay
    byte for byte equal to the training caption (faceswap_caption_v2.py of the dataset builders)."""
    src = "image" if image else "video"
    subs = "\n".join(f"<Subject {i + 1}> is the person whose face is outlined by the thin {c} line in the guide {src}."
                     for i, c in enumerate(names))
    refs = "; ".join(f"<Picture {i + 1}> is the face reference for the {c} line" for i, c in enumerate(names))
    repl = "; ".join(f"replace the face of <Subject {i + 1}> ({c} line) with the face shown in <Picture {i + 1}>"
                     for i, c in enumerate(names))
    moments = ("" if image else " The lines appear only in a few moments of the guide video; between them each subject is "
               "the same person, followed through the whole video wherever they move.")
    motion = "head orientation, gaze and facial expression" if image else "head motion, gaze, blinks, facial expression and lip movements"
    keep = "and the framing" if image else "the camera motion and every cut of the guide video"
    return ("face_swap:\n"
            f"subject_definitions: The guide {src} is the source {src}. Thin colored lines on the inner border of a face mark "
            "the faces to replace; the pixels inside and around the lines provide head pose, expression, gaze, lighting and "
            "scene context. The color of each line is the only link between a face and its reference, wherever the faces "
            f"are placed and whatever they look like.{moments}\n{subs}\n{refs}.\n\n"
            f"summary: [{src} editing + reference generation]\n\n"
            "detailed_description:\n"
            f"In the target {src}, {repl}{'' if image else ', from the first frame to the last'}.\n"
            "Transfer only the facial identity from each assigned reference: eyes, eyebrows, nose, lips, face shape and skin "
            f"tone; do not copy the hair, head shape, pose, expression{'' if image else ', clothing'} or lighting of the reference.\n"
            "Change only the faces inside the lines. For each <Subject>, keep the source hair, hairline and hair color, the "
            f"{motion}, the ears, body and clothing, and adapt each new face to the source lighting, shadows and exposure; "
            "do not mix features between subjects.\n"
            f"Preserve every unmarked person, the background, held objects, foreground occluders{'' if image else ','} {keep}.\n"
            "Remove all colored lines from the result.")


FACE_CAPTIONS = ("v1 (current LoRAs)", "v2 (color-shuffled datasets)")


def face_swap_prompt(subjects, image=False, colors=None, caption=FACE_CAPTIONS[0]):
    """The multi face swap LoRA prompt (sparse face contour marks in the picker): subject j = color j = Picture j.
    colors: optional color names per subject (when the picked colors are not 1..k, Pictures stay 1..k in that order)."""
    subjects = sorted({int(s) for s in subjects if 1 <= int(s) <= 4}) or [1]
    src = "image" if image else "video"
    names = list(colors) if colors else [COLOR_NAMES[j - 1] for j in subjects]; k = len(subjects)
    if str(caption).startswith("v2"):
        return face_swap_prompt_v2(names, image=image)
    subs = "\n".join(f"<Subject {j}> is the person whose face is outlined by the thin {c} line in the guide {src}."
                     for j, c in zip(subjects, names))
    pics = [f"<Picture {j}>" for j in subjects]
    pics = pics[0] if k == 1 else ", ".join(pics[:-1]) + " and " + pics[-1]
    repl = "; ".join(f"replace the face of <Subject {j}> with the face shown in <Picture {j}>" for j in subjects)
    moments = ("" if image else " The lines appear only in a few moments of the guide video; between them each subject is "
               "the same person, followed through the whole video wherever they move.")
    motion = "head orientation, gaze and facial expression" if image else "head motion, gaze, blinks, facial expression and lip movements"
    keep = "and the framing" if image else "the camera motion and every cut of the guide video"
    return ("face_swap:\n"
            f"subject_definitions: The guide {src} is the source {src}. Thin colored lines on the inner border of a face mark "
            "the faces to replace; the pixels inside and around the lines provide head pose, expression, gaze, lighting and "
            f"scene context.{moments}\n{subs}\n{pics} {'is a reference' if k == 1 else 'are references'}.\n\n"
            f"summary: [{src} editing + reference generation]\n\n"
            "detailed_description:\n"
            f"In the target {src}, {repl}{'' if image else ', from the first frame to the last'}.\n"
            "Transfer only the facial identity from each assigned reference: eyes, eyebrows, nose, lips, face shape and skin "
            f"tone; do not copy the hair, head shape, pose, expression{', clothing' if not image else ''} or lighting of the reference.\n"
            f"For each <Subject> in the guide {src}, preserve the source {motion}, the hair, ears, body and clothing, and "
            "adapt each new face to the source lighting, shadows and exposure; do not mix features between subjects.\n"
            f"Preserve every unmarked person, the background, held objects, foreground occluders{',' if not image else ''} {keep}.\n"
            "Remove all colored lines from the result.")


class BFSH3MultiSwapPrompt:
    DESCRIPTION = __doc__.split("\n\n")[0]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "subjects": ("INT", {"default": 2, "min": 1, "max": 6,
                                 "tooltip": "Number of subjects (1..N) when no report is connected."}),
                    "lora": (["auto (from the picker mode)", "multi character swap (hatching)", "multi face swap (video)",
                              "multi face swap (image)"], {"default": "auto (from the picker mode)",
                              "tooltip": "Which LoRA's training prompt to write. auto: face swap when the picker ran in "
                                         "'sparse face contour' mode (image when it had one frame), else character swap."})},
                "optional": {
                    "report": ("STRING", {"forceInput": True,
                               "tooltip": "The Video Segment Picker's report: its subjects with a mask are used."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "BFS/H3"

    def build(self, subjects, lora="auto (from the picker mode)", report=None):
        ids = list(range(1, int(subjects) + 1)); mode = ""; frames = 0
        if report:
            try:
                r = json.loads(report)
                ids = r.get("active_subjects") or r.get("subjects") or ids
                mode = r.get("mode", ""); frames = len(r.get("frames") or [])
            except Exception:
                pass
        if lora.startswith("auto"):
            lora = ("multi face swap (image)" if frames == 1 else "multi face swap (video)") if mode == "sparse face contour" \
                else "multi character swap (hatching)"
        if lora.startswith("multi face swap"):
            return (face_swap_prompt(ids, image=lora.endswith("(image)")),)
        return (multi_swap_prompt(ids),)


NODE_CLASS_MAPPINGS = {"BFSH3MultiSwapPrompt": BFSH3MultiSwapPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSH3MultiSwapPrompt": "BFS H3 Multi Swap Prompt (experimental)"}
