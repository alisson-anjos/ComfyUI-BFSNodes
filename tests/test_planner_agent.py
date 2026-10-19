"""CPU-only agent contract, tools and sampler integration tests; no models or network."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import torch  # import before patch.dict(sys.modules), which must not unload torch extensions
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_bfs_agent_test_package"
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = pkg


def load(name):
    spec = importlib.util.spec_from_file_location(
        PACKAGE + "." + name, ROOT / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


core = load("bfs_planner_agent")
inference = sys.modules[PACKAGE + ".bfs_agent_inference"]
runtime = load("bfs_agent_runtime")
preview = load("bfs_agent_preview")
diagnostics = load("bfs_agent_diagnostics")
tae = load("bfs_agent_tae")


def action(tool, **args):
    return json.dumps({"tool": tool, "arguments": args, "summary": "test"})


class SessionTests(unittest.TestCase):
    def session(self, **settings):
        return core.Session("Transfer motion", {"video": "source.mp4"}, settings)

    def test_reasoning_and_strict_action(self):
        self.assertEqual(
            core.parse_action(
                "<think>private</think>```json\n" + action("inspect_plan") + "\n```"
            )["tool"],
            "inspect_plan",
        )
        for invalid in (
            '{"tool":"shell","arguments":{}}',
            '{"tool":"render","arguments":[]}',
            "prefix " + action("render"),
        ):
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                core.parse_action(invalid)

    def test_invalid_final_action_has_actionable_error(self):
        for text, message in (
            ("", "empty final action"),
            ("I will inspect the video", "valid tool action JSON"),
            ('{"tool":', "valid tool action JSON"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                core.parse_action(text)

    def test_empty_thinking_output_has_actionable_error(self):
        fake = types.ModuleType("comfy_extras.nodes_textgen")
        fake.TextGenerate = object()
        planner = types.SimpleNamespace(
            _VLM={"clip": object()},
            _textgenerate_execute=lambda *a, **kw: types.SimpleNamespace(
                args=("", "private reasoning")
            ),
        )
        session = self.session()
        with patch.dict(sys.modules, {"comfy_extras.nodes_textgen": fake}):
            with self.assertRaisesRegex(
                ValueError, "Thinking off during finalization"
            ) as error:
                runtime.Runtime(planner).decide(session)
        self.assertNotIn("private reasoning", str(error.exception))

    def test_thinking_finalization_respects_total_budget_and_hides_reasoning(self):
        fake = types.ModuleType("comfy_extras.nodes_textgen")
        fake.TextGenerate = object()
        calls = []
        final = action("inspect_source", shot=0)

        def generate(*args, **kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            calls.append(kwargs)
            return (
                types.SimpleNamespace(args=("", "private reasoning"))
                if len(calls) == 1
                else types.SimpleNamespace(args=(final, ""))
            )

        planner = types.SimpleNamespace(
            _VLM={"clip": object()}, _textgenerate_execute=generate
        )
        session = self.session(max_tokens=8192)
        with patch.dict(sys.modules, {"comfy_extras.nodes_textgen": fake}):
            result = runtime.Runtime(planner).decide(session)
        self.assertEqual(core.parse_action(result)["tool"], "inspect_source")
        self.assertEqual([c["thinking"] for c in calls], [True, False])
        self.assertEqual([c["max_length"] for c in calls], [6144, 2048])
        self.assertTrue(session.settings["thinking"])
        self.assertNotIn("private reasoning", calls[1]["prompt"])
        self.assertNotIn("private reasoning", json.dumps(session.public()))
        self.assertEqual(session.events[-1]["phase"], "finalize_action")

    def test_valid_thinking_action_needs_no_finalization(self):
        fake = types.ModuleType("comfy_extras.nodes_textgen")
        fake.TextGenerate = object()
        from unittest.mock import Mock

        generate = Mock(
            return_value=types.SimpleNamespace(args=(action("inspect_plan"), "private"))
        )
        planner = types.SimpleNamespace(
            _VLM={"clip": object()}, _textgenerate_execute=generate
        )
        session = self.session()
        with patch.dict(sys.modules, {"comfy_extras.nodes_textgen": fake}):
            runtime.Runtime(planner).decide(session)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual([e["phase"] for e in session.events], ["thinking"])

    def test_no_second_call_when_thinking_is_disabled(self):
        fake = types.ModuleType("comfy_extras.nodes_textgen")
        fake.TextGenerate = object()
        from unittest.mock import Mock

        generate = Mock(return_value=types.SimpleNamespace(args=("", "")))
        planner = types.SimpleNamespace(
            _VLM={"clip": object()}, _textgenerate_execute=generate
        )
        with patch.dict(sys.modules, {"comfy_extras.nodes_textgen": fake}):
            with self.assertRaisesRegex(ValueError, "with Thinking off"):
                runtime.Runtime(planner).decide(self.session(thinking=False))
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args.kwargs["max_length"], 4096)

    def test_pause_after_thinking_skips_finalization_and_parsing(self):
        fake = types.ModuleType("comfy_extras.nodes_textgen")
        fake.TextGenerate = object()
        session = self.session()
        calls = []

        def generate(*args, **kwargs):
            calls.append(kwargs)
            session.cancel.set()
            session.status = "paused"
            return types.SimpleNamespace(args=("", "private"))

        planner = types.SimpleNamespace(
            _VLM={"clip": object()}, _textgenerate_execute=generate
        )
        with patch.dict(sys.modules, {"comfy_extras.nodes_textgen": fake}):
            session.tick(
                runtime.Runtime(planner).decide,
                lambda *_: self.fail("must not execute"),
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(session.error_count, 0)
        self.assertEqual([e["phase"] for e in session.events], ["thinking"])

    def test_finalization_does_not_bypass_registered_tool_validation(self):
        fake = types.ModuleType("comfy_extras.nodes_textgen")
        fake.TextGenerate = object()
        from unittest.mock import Mock

        generate = Mock(
            side_effect=[
                types.SimpleNamespace(args=("", "private")),
                types.SimpleNamespace(args=(action("shell"), "")),
            ]
        )
        planner = types.SimpleNamespace(
            _VLM={"clip": object()}, _textgenerate_execute=generate
        )
        session = self.session()
        with patch.dict(sys.modules, {"comfy_extras.nodes_textgen": fake}):
            session.tick(
                runtime.Runtime(planner).decide,
                lambda *_: self.fail("must not execute"),
            )
        self.assertIn("registered tool", session.observation["error"])
        self.assertEqual(session.renders, 0)

    def test_plan_is_isolated(self):
        p = {"segs": [{"prompt": "old"}]}
        s = core.Session("task", p)
        s.plan["segs"][0]["prompt"] = "new"
        self.assertEqual(p["segs"][0]["prompt"], "old")

    def test_action_and_render_budgets(self):
        s = self.session(max_actions=1)
        s.tick(lambda _: action("inspect_plan"), lambda *_: ({"ok": True}, []))
        s.tick(lambda _: self.fail("must not decide"), None)
        self.assertEqual(s.status, "budget_exhausted")
        s = self.session(max_renders=1)
        s.tick(lambda _: action("render"), None)
        self.assertEqual(s.status, "waiting_render")
        self.assertEqual(s.renders, 1)
        nonce = s.pending_render["nonce"]
        self.assertTrue(nonce)
        s.status = "running"
        s.tick(lambda _: action("render"), None)
        self.assertIn("budget", s.observation["error"])

    def test_cancel_between_decision_and_tool(self):
        s = self.session()

        def decide(_):
            s.cancel.set()
            s.status = "paused"
            return action("inspect_plan")

        s.tick(decide, lambda *_: self.fail("must not execute"))
        self.assertEqual(s.status, "paused")
        self.assertFalse(s.events)

    def test_lock_prevents_duplicate_actions(self):
        s = self.session()
        s.lock.acquire()
        try:
            with self.assertRaises(ValueError):
                s.tick(None, None)
        finally:
            s.lock.release()

    def test_three_failures_stop_and_final_requires_result(self):
        s = self.session()
        for _ in range(3):
            s.tick(lambda _: action("finish", summary="perfect"), None)
        self.assertEqual(s.status, "error")
        s = self.session()
        s.combined = {"filename": "out.mp4"}
        s.rendered_revision = s.revision
        s.tick(lambda _: action("finish", summary="reviewed"), None)
        self.assertEqual(s.status, "complete")

    def test_persistence_and_restore(self):
        s = self.session()
        s.workflow = {"1": {"class_type": "BFSShotPlanner", "inputs": {}}}
        s.planner_id = "1"
        s.event("action", tool="inspect_plan")
        s.persist(tempfile.gettempdir())
        try:
            d = json.loads((Path(tempfile.gettempdir()) / (s.id + ".json")).read_text())
            self.assertEqual(d["workflow"], s.workflow)
            self.assertEqual(d["events"][0]["tool"], "inspect_plan")
        finally:
            (Path(tempfile.gettempdir()) / (s.id + ".json")).unlink()


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.p = types.SimpleNamespace(
            folder_paths=types.SimpleNamespace(
                get_output_directory=lambda: str(self.root),
                get_input_directory=lambda: str(self.root),
            )
        )
        self.rt = runtime.Runtime(self.p)
        self.s = core.Session("task", {"segs": []})

    def tearDown(self):
        self.tmp.cleanup()

    def test_context_directory_confinement(self):
        (self.root / "notes.md").write_text("motion context")
        out, _ = self.rt.execute(
            self.s, "read_file", {"root": "input", "path": "notes.md"}
        )
        self.assertEqual(out["text"], "motion context")
        self.assertTrue(out["untrusted"])
        for path in ("../outside", "/etc/passwd", ".secrets/key", "api_token.txt"):
            with self.assertRaises(ValueError):
                core.bounded_path({"input": self.root}, "input", path)
        (self.root / "escape").symlink_to("/etc")
        with self.assertRaises(ValueError):
            core.bounded_path({"input": self.root}, "input", "escape/passwd")

    def test_public_web_boundaries(self):
        for url in (
            "file:///etc/passwd",
            "http://example.com",
            "https://user:pass@example.com",
            "https://example.com:8000",
        ):
            with self.assertRaises(ValueError):
                runtime.check_public_url(url)
        for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"):
            with patch.object(
                runtime.socket,
                "getaddrinfo",
                return_value=[(None, None, None, None, (ip, 443))],
            ):
                with self.assertRaises(ValueError):
                    runtime.check_public_url("https://example.com")

    def test_sampler_settings_and_revision(self):
        self.s.workflow = {
            "9": {
                "class_type": "KSampler",
                "inputs": {"seed": 1, "steps": 28, "cfg": 1},
            }
        }
        self.s.reviewed = {0}
        self.s.combined = {"path": "old"}
        out, _ = self.rt.execute(
            self.s,
            "update_sampler",
            {"node_id": "9", "changes": {"seed": 42, "steps": 20}},
        )
        self.assertEqual(self.s.workflow["9"]["inputs"]["seed"], 42)
        self.assertEqual(self.s.revision, 1)
        self.assertFalse(self.s.reviewed)
        self.assertIsNone(self.s.combined)
        for changes in ({"steps": 0}, {"seed": -1}, {"cfg": 99}, {"shell": "oops"}):
            with self.assertRaises(ValueError):
                self.rt.execute(
                    self.s, "update_sampler", {"node_id": "9", "changes": changes}
                )

    def test_split_preserves_settings(self):
        self.s.plan = {
            "segs": [{"prompt": "first"}, {"prompt": "second"}],
            "bounds": [10],
        }
        segs = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
        self.rt.shot = lambda *_: ("v", {}, segs, 0, segs[0])
        self.rt.execute(self.s, "split_shot", {"shot": 0, "frame": 5})
        self.assertEqual(self.s.plan["bounds"], [5, 10])
        self.assertEqual(
            [x["prompt"] for x in self.s.plan["segs"]], ["first", "first", "second"]
        )

    def test_finish_stale_generation_rejected(self):
        self.s.combined = {"path": "old"}
        self.s.rendered_revision = 0
        self.rt.invalidate(self.s)
        self.s.tick(lambda _: action("finish"), None)
        self.assertIn("Combine", self.s.observation["error"])

    def test_sam_dispatch_and_overlay_observation(self):
        self.p.DEFAULT_MASK = {"text": "", "points": [], "key": 0, "padding": 0.15}
        self.p.plan_spec = lambda plan, mask, path: mask
        calls = []
        self.p.mask_preview = lambda path, a, start, length, spec, count: (
            calls.append(spec) or {"frames": [], "coverage": 0.2}
        )
        self.rt.shot = lambda s, args: (
            "video",
            {},
            [{}],
            0,
            {
                "start": 0,
                "end": 20,
                "mask": s.plan.get("segs", [{}])[0].get("mask", {})
                if s.plan["segs"]
                else {},
            },
        )
        out, _ = self.rt.execute(
            self.s,
            "segment",
            {
                "shot": 0,
                "text": "woman in red",
                "points": [{"x": 0.5, "y": 0.4, "label": 1}],
                "key": 4,
            },
        )
        self.assertEqual(out["coverage"], 0.2)
        self.assertEqual(calls[0]["key"], 4)
        self.assertEqual(self.s.plan["segs"][0]["mask"]["text"], "woman in red")


class PreviewTests(unittest.TestCase):
    def test_packed_av_preview_extracts_only_video(self):
        video = torch.arange(24 * 5 * 2 * 2).reshape(1, 24, 5, 2, 2)
        audio = torch.full((1, 32, 2, 7), -99)
        packed = torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
        result = preview.video_latent(packed, [video.shape, audio.shape])
        self.assertTrue(torch.equal(result, video))
        self.assertEqual(tuple(packed.shape), (1, 1, video.numel() + audio.numel()))
        with self.assertRaisesRegex(ValueError, "no latent_shapes"):
            preview.video_latent(packed)
        with self.assertRaisesRegex(ValueError, "does not match"):
            preview.video_latent(packed, [video.shape])

    def test_nested_and_dense_h3_video_preview(self):
        video = torch.ones(1, 24, 3, 2, 2)
        nested = types.SimpleNamespace(
            is_nested=True, unbind=lambda: [video, torch.zeros(1, 32, 2, 7)]
        )
        self.assertIs(preview.video_latent(nested), video)
        self.assertIs(preview.video_latent(video), video)
        self.assertTrue(torch.equal(preview.video_latent(video[0]), video))
        with self.assertRaisesRegex(ValueError, "Expected H3 video"):
            preview.video_latent(torch.zeros(1, 16, 3, 2, 2))

    def test_packed_tiny_vae_uses_temporal_neighborhoods(self):
        self.run_preview_case()

    def test_explicit_preview_retry_still_interrupts_generation(self):
        self.run_preview_case(retry=True)

    def test_full_vae_uses_single_latent_snapshots_and_preserves_callback(self):
        self.run_preview_case(full_vae=True)

    def test_preview_shape_failure_preserves_generation(self):
        self.run_preview_case(bad_shape=True)

    def test_preview_decode_failure_preserves_generation(self):
        self.run_preview_case(bad_decode=True)

    def test_fatal_preview_cuda_error_skips_model_reload(self):
        self.run_preview_case(fatal_cuda=True)

    def run_preview_case(
        self,
        full_vae=False,
        bad_shape=False,
        bad_decode=False,
        retry=False,
        fatal_cuda=False,
    ):
        sid = "a" * 32
        session = core.Session("task", {}, session_id=sid)
        session.preview = {
            "review": {"decision": "retry"},
            "frames": [{"src": "old render"}],
        }
        runtime.SESSIONS[sid] = session
        self.addCleanup(runtime.SESSIONS.pop, sid, None)
        p = types.ModuleType(PACKAGE + ".bfs_shot_loop")
        p._notify = lambda *a: None
        p._VLM = {"clip": object()}
        p._parse_json = json.loads
        p._textgenerate_execute = lambda *a, **kw: types.SimpleNamespace(
            args=(json.dumps({"decision": "retry", "reason": "major failure"}),)
        )
        mm = types.ModuleType("comfy.model_management")
        mm.loaded_models = lambda **kw: ["diffusion"]
        restored = []
        mm.load_models_gpu = lambda models: restored.append(models)
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.model_management = mm
        video = torch.ones(1, 24, 5, 2, 2)
        audio = torch.full((1, 32, 2, 7), 99.0)
        packed = torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
        shapes = None if bad_shape else [video.shape, audio.shape]
        model = types.SimpleNamespace(
            inner_model=types.SimpleNamespace(latent_shapes=shapes),
            model_patcher=types.SimpleNamespace(
                get_model_object=lambda _: types.SimpleNamespace(
                    process_out=lambda x: x * 2
                )
            ),
        )
        decoded = []

        def decode(z):
            decoded.append(z.clone())
            if fatal_cuda:
                raise RuntimeError(
                    "CUDA error: an illegal memory access was encountered"
                )
            if bad_decode:
                raise ValueError("decoder failed")
            return torch.zeros(1, 8, 8, 3)

        stage = type("MiniMaxH3VideoVAE" if full_vae else "TinyVAE", (), {})()
        vae = types.SimpleNamespace(first_stage_model=stage, decode=decode)
        callbacks = []

        def sample(m, sigmas, args, cb, *rest):
            for step in range(3):
                cb(step, packed, packed * 0, 3)
            return "generated video"

        with patch.dict(
            sys.modules, {"comfy": comfy, "comfy.model_management": mm, p.__name__: p}
        ):
            fake = types.ModuleType("comfy_extras.nodes_textgen")
            fake.TextGenerate = object()
            wrapper = preview.PreviewSampler(
                types.SimpleNamespace(sample=sample), sid, 1, 0, 3, retry, vae
            )
            if fatal_cuda:
                with patch.object(inference, "CUDA_FAULTED", False):
                    with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
                        wrapper.sample(model, [1, 0.5, 0.1, 0], {}, None, packed)
                    self.assertTrue(inference.CUDA_FAULTED)
                self.assertEqual(restored, [])
                return
            if retry:
                with patch.dict(sys.modules, {"comfy_extras.nodes_textgen": fake}):
                    with self.assertRaises(preview.PreviewRetry):
                        wrapper.sample(model, [1, 0.5, 0.1, 0], {}, None, packed)
                self.assertEqual(len(restored), 2)
                return
            result = wrapper.sample(
                model, [1, 0.5, 0.1, 0], {}, lambda *a: callbacks.append(a), packed
            )
        self.assertEqual(result, "generated video")
        self.assertEqual(len(callbacks), 3)
        self.assertTrue(all(c[1] is packed for c in callbacks))
        if bad_shape or bad_decode:
            self.assertIn("error", session.preview)
            self.assertNotIn("review", session.preview)
            self.assertEqual(
                len([e for e in session.events if e["kind"] == "preview_error"]), 1
            )
            self.assertEqual(len(restored), 1)
        else:
            if full_vae:
                self.assertTrue(all(z.shape[2] == 1 for z in decoded))
            else:
                self.assertTrue(all(1 < z.shape[2] <= 3 for z in decoded))
            self.assertTrue(all(torch.all(z == 2) for z in decoded))
            self.assertEqual(
                session.preview["decode_mode"],
                "single-latent snapshots" if full_vae else "temporal neighborhoods",
            )
            self.assertEqual(len(restored), 3)

    def test_inactive_session_is_exact_passthrough(self):
        p = types.ModuleType(PACKAGE + ".bfs_shot_loop")
        sys.modules[p.__name__] = p
        mm = types.ModuleType("comfy.model_management")
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.model_management = mm
        calls = []
        sampler = types.SimpleNamespace(
            sample=lambda *args: calls.append(args) or "result"
        )
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm}):
            wrapped = preview.PreviewSampler(sampler, "missing", 5, 15, 4, True)
            result = wrapped.sample("model", [1, 0], {}, None, "noise")
        self.assertEqual(result, "result")
        self.assertEqual(calls[0][0], "model")

    def test_x0_preview_callback_restores_models(self):
        sid = "f" * 32
        s = core.Session("task", {}, session_id=sid)
        runtime.SESSIONS[sid] = s
        events = []
        p = types.ModuleType(PACKAGE + ".bfs_shot_loop")
        p._notify = lambda *a: events.append(a)
        mm = types.ModuleType("comfy.model_management")
        mm.loaded_models = lambda **kw: ["diffusion"]
        restored = []
        mm.load_models_gpu = lambda models: restored.append(models)
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.model_management = mm
        x0 = torch.ones(1, 24, 5, 2, 2)
        noise = torch.zeros_like(x0)

        def sample(m, sigmas, args, cb, *rest):
            for step in range(3):
                cb(step, x0, noise, 3)
            return "done"

        z = []
        vae = types.SimpleNamespace(
            first_stage_model=object(),
            decode=lambda x: z.append(x.clone()) or torch.zeros(3, 8, 8, 3),
        )
        model = types.SimpleNamespace(
            model_patcher=types.SimpleNamespace(
                get_model_object=lambda _: types.SimpleNamespace(
                    process_out=lambda x: x * 2
                )
            )
        )
        with patch.dict(
            sys.modules, {"comfy": comfy, "comfy.model_management": mm, p.__name__: p}
        ):
            wrapped = preview.PreviewSampler(
                types.SimpleNamespace(sample=sample), sid, 2, 0, 3, False, vae
            )
            self.assertEqual(
                wrapped.sample(model, [1, 0.5, 0.1, 0], {}, None, noise), "done"
            )
        self.assertEqual(s.preview["step"], 3)
        self.assertEqual(len(s.preview["frames"]), 3)
        self.assertTrue(all(torch.all(x == 2) for x in z))
        self.assertEqual(len(restored), 2)
        runtime.SESSIONS.pop(sid)


class ServerLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_server_cycle_without_browser_requeue(self):
        await self.run_server_cycle()

    async def test_invalid_vae_render_stops_before_another_vlm_decision(self):
        await self.run_server_cycle(invalid_vae=True)

    async def run_server_cycle(self, invalid_vae=False):
        import asyncio
        from collections import deque

        with tempfile.TemporaryDirectory() as directory:
            registered = {}

            class Routes:
                def post(self, path):
                    return lambda fn: registered.setdefault(("POST", path), fn)

                def get(self, path):
                    return lambda fn: registered.setdefault(("GET", path), fn)

            class Queue:
                def __init__(self):
                    self.history = {}
                    self.graphs = []

                def put(self, item):
                    self.graphs.append(item[2])
                    is_context = any(
                        n.get("class_type") == "BFSAgentContext"
                        for n in item[2].values()
                    )
                    self.history[item[1]] = {
                        "status": (
                            {
                                "status_str": "error",
                                "messages": [
                                    [
                                        "execution_error",
                                        {
                                            "exception_message": "ERROR: VAE is invalid: None"
                                        },
                                    ]
                                ],
                            }
                            if invalid_vae and not is_context
                            else {"status_str": "success"}
                        )
                    }

                def get_history(self, prompt_id):
                    return (
                        {prompt_id: self.history[prompt_id]}
                        if prompt_id in self.history
                        else {}
                    )

                def get_current_queue(self):
                    return [], []

            queue = Queue()
            server = types.SimpleNamespace(
                routes=Routes(), prompt_queue=queue, number=0
            )
            server_module = types.ModuleType("server")
            server_module.PromptServer = types.SimpleNamespace(instance=server)
            execution = types.ModuleType("execution")

            async def validate(prompt_id, graph, targets):
                return True, None, list(graph), {}

            execution.validate_prompt = validate
            execution.SENSITIVE_EXTRA_DATA_KEYS = []
            nodes = types.ModuleType("nodes")
            nodes.NODE_CLASS_MAPPINGS = {
                "SaveVideo": types.SimpleNamespace(OUTPUT_NODE=True)
            }
            p = types.SimpleNamespace(
                folder_paths=types.SimpleNamespace(
                    get_output_directory=lambda: directory
                ),
                _load_plan=json.loads,
                _notify=lambda *_: None,
            )
            sid = None
            with patch.dict(
                sys.modules,
                {"server": server_module, "execution": execution, "nodes": nodes},
            ):
                rt = runtime.register(p)
                script = deque(
                    [
                        action("inspect_plan"),
                        action("render"),
                        action("inspect_result", shot=0),
                        action("combine"),
                        action("finish", summary="done"),
                    ]
                )
                worker_threads = set()

                def decide(s):
                    worker_threads.add(__import__("threading").get_ident())
                    self.assertTrue(torch.is_inference_mode_enabled())
                    return script.popleft()

                rt.decide = decide

                def execute(s, tool, args):
                    if tool == "inspect_result":
                        s.reviewed.add(0)
                    if tool == "combine":
                        s.combined = {"filename": "final.mp4"}
                    return {"tool": tool}, []

                rt.execute = execute
                graph = {
                    "1": {
                        "class_type": "BFSShotPlanner",
                        "inputs": {"vlm": ["2", 0], "plan": "{}"},
                    },
                    "2": {"class_type": "CLIPLoader", "inputs": {}},
                    "3": {"class_type": "SaveVideo", "inputs": {"images": ["1", 0]}},
                    "4": {"class_type": "SaveVideo", "inputs": {}},
                }
                body = {
                    "task": "motion transfer",
                    "plan": {"video": "test.mp4"},
                    "workflow": graph,
                    "planner_id": "1",
                }
                request = types.SimpleNamespace(
                    json=lambda: asyncio.sleep(0, result=body)
                )
                response = await registered[("POST", "/bfs/shotloop/agent/start")](
                    request
                )
                self.assertEqual(response.status, 200, response.text)
                sid = json.loads(response.text)["id"]
                s = runtime.SESSIONS[sid]
                for _ in range(250):
                    if s.status in ("complete", "error", "budget_exhausted"):
                        break
                    await asyncio.sleep(0.01)
                if invalid_vae:
                    self.assertEqual(s.status, "error")
                    self.assertIn("VAE loader rejected", s.observation["error"])
                    self.assertTrue(s.requires_new_task)
                    response = await registered[
                        ("POST", "/bfs/shotloop/agent/control")
                    ](
                        types.SimpleNamespace(
                            json=lambda: asyncio.sleep(
                                0, result={"id": sid, "action": "resume"}
                            )
                        )
                    )
                    self.assertEqual(response.status, 400)
                    self.assertIn("Start automatic task", response.text)
                    self.assertEqual(s.status, "error")
                    self.assertEqual(
                        len(script), 3
                    )  # no inspect_result/combine/finish decisions after loader failure
                    self.assertEqual(len(queue.graphs), 2)
                    runtime.SESSIONS.pop(sid)
                    return
                self.assertEqual(s.status, "complete", s.observation)
                self.assertEqual(len(worker_threads), 1)
                context = next(
                    e["result"]
                    for e in s.events
                    if e["kind"] == "observation"
                    and e.get("result", {}).get("kind") == "context"
                )
                self.assertFalse(context["current_plan_rendered"])
                self.assertIn("No video was generated", context["message"])
                self.assertEqual(len(queue.graphs), 2)
                self.assertEqual(s.renders, 1)
                self.assertTrue(
                    any(
                        x["class_type"] == "BFSAgentContext"
                        for x in queue.graphs[0].values()
                    )
                )
                self.assertNotIn("3", queue.graphs[0])
                self.assertNotIn("4", queue.graphs[1])
                self.assertEqual(
                    json.loads(queue.graphs[1]["1"]["inputs"]["plan"])["run"],
                    "recursive",
                )
                self.assertEqual(s.rendered_revision, s.revision)
                s.status = "paused"
                controls = {
                    "id": sid,
                    "action": "resume",
                    "settings": {"thinking": False, "web": False, "max_tokens": 8192},
                }
                control_response = await registered[
                    ("POST", "/bfs/shotloop/agent/control")
                ](types.SimpleNamespace(json=lambda: asyncio.sleep(0, result=controls)))
                self.assertEqual(control_response.status, 200, control_response.text)
                self.assertFalse(s.settings["thinking"])
                self.assertFalse(s.settings["web"])
                self.assertEqual(s.settings["max_tokens"], 8192)
                s.cancel.set()
                s.status = "stopped"
                runtime.SESSIONS.pop(sid)


class VLMDiagnosticsTests(unittest.TestCase):
    settings = {"max_tokens": 4096, "thinking": True}

    def clip(self):
        config = types.SimpleNamespace(
            max_position_embeddings=32768,
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        model_type = type("Llama2_", (), {"__module__": "comfy.text_encoders.llama"})
        model = model_type()
        model.config = config
        transformer = types.SimpleNamespace(
            model_type="qwen3vl_8b", model=model, visual=object()
        )
        tokenizer = types.SimpleNamespace(
            tokenize_with_weights=lambda text, thinking=False: {}
        )
        patcher = types.SimpleNamespace(
            load_device=torch.device("cuda:0"),
            model_size=lambda: 4 * 1024**3,
            loaded_size=lambda: 0,
        )
        return types.SimpleNamespace(
            cond_stage_model=types.SimpleNamespace(modules=lambda: [transformer]),
            tokenizer=tokenizer,
            patcher=patcher,
            generate=lambda: None,
        )

    def test_unknown_model_does_not_invent_capabilities(self):
        report = diagnostics.inspect_vlm(object(), self.settings)
        self.assertEqual(report["images"], "unknown")
        self.assertIsNone(report["context_tokens"])
        self.assertIsNone(report["kv_estimate"])

    def test_model_config_and_tokenizer_do_not_prove_checkpoint_thinking(self):
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            report = diagnostics.inspect_vlm(
                self.clip(), self.settings, "example.safetensors"
            )
        self.assertEqual(report["context_tokens"], 32768)
        self.assertEqual(report["images"], "vision encoder detected")
        self.assertIn("supports", report["thinking_control"])
        self.assertIn("unknown", report["thinking_checkpoint"])
        self.assertIsNone(report["gpu"])

    def test_kv_estimate_subtracts_unloaded_weights_and_reserve(self):
        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(
                torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3)
            ),
            patch.object(torch.cuda, "get_device_name", return_value="Test GPU"),
        ):
            report = diagnostics.inspect_vlm(self.clip(), self.settings)
        reserve = int(16 * 1024**3 * 0.15)
        expected = (8 * 1024**3 - 4 * 1024**3 - reserve) // (2 * 32 * 8 * 128 * 4)
        self.assertEqual(report["kv_estimate"]["total_tokens"], expected)
        self.assertIn("Not a guaranteed fit", report["kv_estimate"]["assumptions"])

    def test_no_capacity_when_weights_use_available_memory(self):
        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(
                torch.cuda, "mem_get_info", return_value=(2 * 1024**3, 16 * 1024**3)
            ),
            patch.object(torch.cuda, "get_device_name", return_value="Test GPU"),
        ):
            report = diagnostics.inspect_vlm(self.clip(), self.settings)
        self.assertEqual(report["kv_estimate"]["total_tokens"], 0)

    def test_hybrid_cache_is_not_estimated_as_full_attention(self):
        clip = self.clip()
        clip.cond_stage_model.modules()[0].model = types.SimpleNamespace(
            config=clip.cond_stage_model.modules()[0].model.config
        )
        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(
                torch.cuda, "mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3)
            ),
            patch.object(torch.cuda, "get_device_name", return_value="Test GPU"),
        ):
            report = diagnostics.inspect_vlm(clip, self.settings)
        self.assertIsNone(report["kv_estimate"])


class InferenceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fault_patch = patch.object(inference, "CUDA_FAULTED", False)
        self.fault_patch.start()
        self.addCleanup(self.fault_patch.stop)

    def test_inference_context_and_cleanup_for_each_call(self):
        records = []

        def work():
            records.append(
                ("work", torch.is_inference_mode_enabled(), torch.is_grad_enabled())
            )
            return "result"

        def cleanup():
            records.append(("cleanup", torch.is_inference_mode_enabled()))

        original_mode = torch.is_inference_mode_enabled()
        with patch.object(inference, "cleanup_inference", cleanup):
            self.assertEqual(inference.run_inference(work), "result")
            self.assertEqual(inference.run_inference(work), "result")
        self.assertEqual(
            records,
            [
                ("work", True, False),
                ("cleanup", True),
                ("work", True, False),
                ("cleanup", True),
            ],
        )
        self.assertEqual(torch.is_inference_mode_enabled(), original_mode)

    def test_ordinary_error_cleans_up_but_does_not_poison_context(self):
        from unittest.mock import Mock

        cleanup = Mock()

        def work():
            raise ValueError("invalid JSON")

        with patch.object(inference, "cleanup_inference", cleanup):
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                inference.run_inference(work)
        cleanup.assert_called_once()
        self.assertFalse(inference.CUDA_FAULTED)

    def test_fatal_cuda_error_blocks_retry_and_skips_allocator_cleanup(self):
        from unittest.mock import Mock

        cleanup = Mock()

        def work():
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        with patch.object(inference, "cleanup_inference", cleanup):
            with self.assertRaisesRegex(RuntimeError, "illegal memory"):
                inference.run_inference(work)
            with self.assertRaisesRegex(RuntimeError, "restart ComfyUI"):
                inference.run_inference(lambda: self.fail("must not retry GPU"))
        cleanup.assert_not_called()
        self.assertTrue(inference.CUDA_FAULTED)

    def test_session_does_not_retry_cuda_fault_three_times(self):
        session = core.Session("task", {})

        def work(_):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        session.tick(work, None)
        self.assertEqual(session.status, "error")
        self.assertEqual(session.error_count, 1)
        self.assertTrue(session.observation["restart_required"])

    def test_dynamic_cleanup_matches_native_order_without_changing_settings(self):
        calls = []
        modules = {
            "comfy.memory_management": types.SimpleNamespace(aimdo_enabled=True),
            "comfy.model_prefetch": types.SimpleNamespace(
                cleanup_prefetch_queues=lambda: calls.append("prefetch")
            ),
            "comfy.model_management": types.SimpleNamespace(
                reset_cast_buffers=lambda: calls.append("cast")
            ),
            "comfy_aimdo.model_vbar": types.SimpleNamespace(
                vbars_reset_watermark_limits=lambda: calls.append("vbar")
            ),
        }
        with patch.dict(sys.modules, modules):
            inference.cleanup_inference()
        self.assertEqual(calls, ["prefetch", "cast", "vbar"])
        self.assertTrue(modules["comfy.memory_management"].aimdo_enabled)


class TinyPreviewLoaderTests(unittest.TestCase):
    def test_kijai_2d_format_is_not_confused_with_temporal_vae(self):
        self.assertTrue(
            tae.is_2d_h3(
                {
                    "1.weight": torch.empty(96, 24, 3, 3),
                    "23.weight": torch.empty(3, 64, 3, 3),
                }
            )
        )
        self.assertFalse(tae.is_2d_h3({"decoder.1.weight": torch.empty(256, 24, 3, 3)}))
        self.assertFalse(
            tae.is_2d_h3(
                {
                    "1.weight": torch.empty(64, 4, 3, 3),
                    "23.weight": torch.empty(3, 64, 3, 3),
                }
            )
        )

    def test_partial_checkpoint_is_rejected_before_decode(self):
        with self.assertRaises(RuntimeError):
            tae.build_h3_decoder({"1.weight": torch.empty(96, 24, 3, 3)})

    def test_tiny_adapter_decodes_one_image_at_a_time_and_clamps_output(self):
        calls = []
        mm = types.ModuleType("comfy.model_management")
        mm.load_models_gpu = lambda models: None
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.model_management = mm
        vae = tae.H3TinyPreviewVAE.__new__(tae.H3TinyPreviewVAE)
        vae.device, vae.dtype, vae.patcher = (
            torch.device("cpu"),
            torch.float32,
            object(),
        )

        def decoder(z):
            calls.append(tuple(z.shape))
            return torch.full((1, 3, 32, 48), 2.0)

        vae.first_stage_model = decoder
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm}):
            output = vae.decode(torch.zeros(1, 24, 2, 2, 3))
        self.assertEqual(calls, [(1, 24, 2, 3)] * 2)
        self.assertEqual(tuple(output.shape), (2, 32, 48, 3))
        self.assertTrue(torch.all(output == 1))
        self.assertTrue(vae.single_latent_preview)

    def test_invalid_vae_loader_failure_is_a_configuration_error(self):
        failure = {
            "status_str": "error",
            "messages": [
                [
                    "execution_error",
                    {
                        "node_type": "VAELoader",
                        "exception_message": "ERROR: VAE is invalid: None",
                    },
                ]
            ],
        }
        self.assertTrue(runtime.invalid_vae_failure(failure))
        self.assertFalse(runtime.invalid_vae_failure({"status_str": "success"}))
        self.assertFalse(
            runtime.invalid_vae_failure({"status_str": "error", "messages": []})
        )


class PreviewHistoryTests(unittest.TestCase):
    def test_capture_survives_restore_and_failure_keeps_review(self):
        with tempfile.TemporaryDirectory() as folder:
            planner = types.SimpleNamespace(
                folder_paths=types.SimpleNamespace(get_output_directory=lambda: folder),
                _notify=lambda *args: None,
            )
            rt = runtime.Runtime(planner)
            session = core.Session("swap", {}, {})
            session.renders = 1
            session.pending_render = {"prompt_id": "test-prompt", "kind": "render"}
            rt.attach(session)
            image = runtime.image_url(torch.zeros(1, 8, 8, 3))
            capture = {
                "shot": 0,
                "step": 7,
                "total_steps": 8,
                "fraction": 0.875,
                "approximate": True,
                "frames": [{"src": image}],
            }
            session.record_preview(capture)
            files = list((Path(folder) / "bfs_agent" / session.id).rglob("*.jpg"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].read_bytes().startswith(b"\xff\xd8"))
            saved = json.loads(
                (Path(folder) / "bfs_agent" / (session.id + ".json")).read_text()
            )
            self.assertNotIn("base64", json.dumps(saved["preview_history"]))
            capture["review"] = {"decision": "retry", "reason": "Wrong composition"}
            session.record_preview(capture)
            self.assertEqual(len(session.preview_history[0]["captures"]), 1)
            rt.render_outcome(session, {"status_str": "error"})
            self.assertEqual(session.preview_history[0]["status"], "visually_rejected")
            restored = rt.get(session.id)
            try:
                self.assertEqual(
                    restored.preview_history[0]["reason"], "Wrong composition"
                )
            finally:
                runtime.SESSIONS.pop(session.id, None)

    def test_failure_without_preview_does_not_invent_images(self):
        with tempfile.TemporaryDirectory() as folder:
            rt = runtime.Runtime(
                types.SimpleNamespace(
                    folder_paths=types.SimpleNamespace(
                        get_output_directory=lambda: folder
                    )
                )
            )
            session = core.Session("swap", {}, {})
            session.pending_render = {"prompt_id": "failed", "kind": "render"}
            rt.render_outcome(session, {"status_str": "error"})
            self.assertEqual(session.preview_history[0]["status"], "technical_failure")
            self.assertEqual(session.preview_history[0]["captures"], [])

    def test_public_explanation_is_bounded_and_private_output_is_discarded(self):
        result = core.parse_action(
            "<think>private reasoning</think>"
            + json.dumps(
                {
                    "tool": "inspect_source",
                    "arguments": {"shot": 0},
                    "observed": "Source not inspected",
                    "reason": "Check movement",
                    "expected": "x" * 900,
                    "uncertainty": "Identity unverified",
                }
            )
        )
        self.assertEqual(result["reason"], "Check movement")
        self.assertEqual(len(result["expected"]), 500)
        self.assertNotIn("private reasoning", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
