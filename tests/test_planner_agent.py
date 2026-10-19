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
runtime = load("bfs_agent_runtime")
preview = load("bfs_agent_preview")
diagnostics = load("bfs_agent_diagnostics")


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
            with self.assertRaisesRegex(ValueError, "thinking output only") as error:
                runtime.Runtime(planner).decide(session)
        self.assertNotIn("private reasoning", str(error.exception))

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
                    self.history[item[1]] = {"status": {"status_str": "success"}}

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
                rt.decide = lambda s: script.popleft()

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
                self.assertEqual(s.status, "complete", s.observation)
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


if __name__ == "__main__":
    unittest.main()
