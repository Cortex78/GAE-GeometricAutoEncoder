"""Exercise the evaluation paths without loading GPU models or checkpoints."""

import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/eval/eval_generation.py"
TREE = ast.parse(SOURCE.read_text())
POLICY = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "_da3_direct_cls_mode"
)
MAIN = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "main"
)
DIRECT = next(
    node for node in ast.walk(MAIN)
    if isinstance(node, ast.If)
    and ast.unparse(node.test).startswith("has_dpt and ")
    and "latent_backend == 'da3_direct'" in ast.unparse(node.test)
)


def run_nodes(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)


class DirectCLSTest(unittest.TestCase):
    def test_cls_policy(self):
        namespace = {"os": os}
        run_nodes([POLICY], namespace)
        for setting, expected in ((None, "zero"), ("zero", "zero"),
                                  ("gt", "gt"), (" GT ", "gt")):
            with self.subTest(setting=setting), patch.dict(os.environ, {}, clear=True):
                if setting is not None:
                    os.environ["DA3_DIRECT_DUMP_CLS"] = setting
                self.assertEqual(namespace["_da3_direct_cls_mode"](), expected)
        for invalid in ("", "gtt", "zeros"):
            with self.subTest(invalid=invalid), patch.dict(
                os.environ, {"DA3_DIRECT_DUMP_CLS": invalid}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "DA3_DIRECT_DUMP_CLS"):
                    namespace["_da3_direct_cls_mode"]()

    def test_geometry_artifacts_and_timing_share_cls(self):
        for cls_mode in ("zero", "gt"):
            for mode in ("generate", "recon"):
                for timing, artifacts, dump in (
                    (False, True, True),   # visualization and geometry together
                    (True, False, False),  # timing only
                    (False, False, True),  # metrics-only geometry dump
                    (False, False, False), # no geometry consumer
                ):
                    with self.subTest(cls=cls_mode, mode=mode, timing=timing,
                                      artifacts=artifacts, dump=dump):
                        self.check_paths(cls_mode, mode, timing, artifacts, dump)

    def check_paths(self, cls_mode, mode, timing, artifacts, dump):
        codec = MagicMock()
        codec.level = 0
        codec.decode_depth.return_value = {}
        decode_dpt = MagicMock(return_value={})
        save = MagicMock()
        torch = SimpleNamespace(
            amp=SimpleNamespace(autocast=lambda *a, **kw: nullcontext()),
            from_numpy=MagicMock(),
        )
        namespace = dict(
            args=SimpleNamespace(timing_json=timing, dump_geometry=dump, mode=mode),
            actual_v=2, pose_list=[0, 1], intri_list=[0, 1],
            direct_cls_mode=cls_mode, save_artifacts=artifacts,
            vae=codec, rae=codec.rae, torch=torch, imgs=MagicMock(),
            imgs_01=MagicMock(), z_sampled=MagicMock(), device="test", H=28, W=28,
            _primary_output_timing=nullcontext, decode_dpt=decode_dpt,
            has_rgb=False, np=MagicMock(), intrinsic_to_K=MagicMock(),
            scene_cond=1, _save_geom_npz=save, os=os, ds_out_dir="test", s_idx=0,
            print=MagicMock(),
        )
        run_nodes(DIRECT.body, namespace)

        needs_cls = cls_mode == "gt" and (timing or artifacts or dump)
        self.assertEqual(codec.encode_all.call_count, int(needs_cls))
        expected_cls = (
            codec.encode_all.return_value[0][:, 0].float() if needs_cls else None
        )
        self.assertEqual(codec.decode_depth.call_count, int(timing or artifacts))
        if timing or artifacts:
            self.assertIs(codec.decode_depth.call_args.args[1], expected_cls)

        propagate = codec.rae.propagate_features
        self.assertEqual(propagate.call_count, (2 if mode == "recon" else 1) if dump else 0)
        for call in propagate.call_args_list:
            self.assertIs(call.kwargs["cls_token"], expected_cls)
        self.assertEqual(decode_dpt.call_count, propagate.call_count)
        self.assertEqual(save.call_count, int(dump))


if __name__ == "__main__":
    unittest.main()
