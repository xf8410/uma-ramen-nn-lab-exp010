"""把多个 model.onnx 合成一个集成模型：仿射校正后逐元素平均。

用法：``python3 compose_ensemble.py --output ens.onnx --models a.onnx b.onnx [...]``

## 为什么必须做仿射校正（写死在此，勿改）

输出 245 维布局冻结（scripts/ramen_nn/model.py）：policy[0:234] + choice[234:242] + value[242:245]。
value 三路是**归一化后**的值，归一化参数由 `fit_value_normalization` 在**训练划分**上拟合，
而 train.py 的 `--seed` 同时决定 split_seed 与 init_seed——EXP-009 三个种子各用不同 seed，
于是**三个模型的 value 头处在三个不同的仿射空间**。直接平均 value 等于把不同单位的数相加。

policy/choice 无归一化，直接平均（logit 平均）即可。

校正做法：对每个成员做一次逐元素仿射，把整段 245 维映射到参考模型（第一个）的坐标系，再平均：

    adj_i = out_i * m_i + b_i
    m_i = [1]*242 + [scale_i / scale_0]
    b_i = [0]*242 + [(center_i - center_0) / scale_0]
    out  = mean_i(adj_i)

前 242 维乘 1 加 0（原样平均），后 3 维先反归一化到原始分、再按参考归一化重新归一化。
这样集成输出与成员同处一个归一化空间，Rust 侧只读 sidecar 里参考模型的
value_normalization，**无需任何改动**。

## 算子与校验

只引入 Mul/Add/Div + 常量 initializer，全部在 export_onnx.py 的保守白名单与 tract 支持集内。
合成后：onnx.checker 结构校验 + onnxruntime 与「逐成员跑一遍再手工平均」对拍（batch=1 与 7），
误差 >=1e-4 即失败；成员 model_config 不一致直接拒绝（结构不同不配谈平均）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POLICY_DIM = 234
CHOICE_DIM = 8
VALUE_DIM = 3
VALUE_BASE = POLICY_DIM + CHOICE_DIM  # 242
OUTPUT_DIM = 245
INPUT_DIM = 754
TOLERANCE = 1e-4


def _sidecar(onnx_path: Path) -> Path:
    """成员/集成的元数据旁路文件路径。"""

    return onnx_path.with_suffix(onnx_path.suffix + ".json")


def _load_member(onnx_path: Path):
    """加载一个成员 ONNX 及其 sidecar，并校验冻结的输入/输出契约。"""

    import onnx

    meta_path = _sidecar(onnx_path)
    if not meta_path.is_file():
        raise SystemExit(f"缺 sidecar: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = onnx.load(str(onnx_path))
    graph = model.graph
    if len(graph.input) != 1 or graph.input[0].name != "input":
        raise SystemExit(f"{onnx_path}: 期望唯一输入 'input'，实得 {[i.name for i in graph.input]}")
    if len(graph.output) != 1 or graph.output[0].name != "output":
        raise SystemExit(f"{onnx_path}: 期望唯一输出 'output'，实得 {[o.name for o in graph.output]}")
    in_dims = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    out_dims = [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]
    if not in_dims or in_dims[-1] != INPUT_DIM:
        raise SystemExit(f"{onnx_path}: 输入维不符 {in_dims}")
    if not out_dims or out_dims[-1] != OUTPUT_DIM:
        raise SystemExit(f"{onnx_path}: 输出维不符 {out_dims}")
    if int(meta.get("input_dim", INPUT_DIM)) != INPUT_DIM or int(meta.get("output_dim", OUTPUT_DIM)) != OUTPUT_DIM:
        raise SystemExit(f"{onnx_path}: sidecar 维度字段不符")
    if "value_normalization" not in meta or "model_config" not in meta:
        raise SystemExit(f"{onnx_path}: sidecar 缺 value_normalization/model_config")
    return model, meta


def _prefixed(model, prefix: str):
    """给成员图内所有名字加前缀（图输入仍叫 input），返回 (nodes, initializers, 输出张量名)。"""

    graph = model.graph
    for node in graph.node:
        node.name = prefix + node.name
        node.input[:] = [prefix + item if item else item for item in node.input]
        node.output[:] = [prefix + item if item else item for item in node.output]
    for init in graph.initializer:
        init.name = prefix + init.name
    for init in graph.sparse_initializer:
        init.name = prefix + init.name
    del graph.value_info[:]
    # 成员自己的图输入不加前缀：把 prefix+"input" 改回 "input"
    for node in graph.node:
        node.input[:] = ["input" if item == prefix + "input" else item for item in node.input]
    return list(graph.node), list(graph.initializer), prefix + "output"


def _affine(meta: dict, ref_meta: dict):
    """返回该成员的 245 维乘/加向量（把 value 头映射到参考坐标系）。"""

    center = np.asarray(meta["value_normalization"]["center"], dtype=np.float64)
    scale = np.asarray(meta["value_normalization"]["scale"], dtype=np.float64)
    ref_center = np.asarray(ref_meta["value_normalization"]["center"], dtype=np.float64)
    ref_scale = np.asarray(ref_meta["value_normalization"]["scale"], dtype=np.float64)
    if center.shape != (VALUE_DIM,) or scale.shape != (VALUE_DIM,):
        raise SystemExit(f"value_normalization 形状异常: center={center.shape} scale={scale.shape}")
    if np.any(ref_scale <= 0) or np.any(scale <= 0):
        raise SystemExit("value_normalization scale 必须为正")
    mul = np.ones(OUTPUT_DIM, dtype=np.float32)
    add = np.zeros(OUTPUT_DIM, dtype=np.float32)
    mul[VALUE_BASE:] = (scale / ref_scale).astype(np.float32)
    add[VALUE_BASE:] = ((center - ref_center) / ref_scale).astype(np.float32)
    return mul, add


def compose(model_paths: list[Path], output_path: Path) -> float:
    """合成集成模型并落盘（含 sidecar），返回对拍最大误差。"""

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if len(model_paths) < 2:
        raise SystemExit("集成至少需要 2 个成员")
    members = [_load_member(path) for path in model_paths]
    configs = [meta["model_config"] for _, meta in members]
    if any(config != configs[0] for config in configs[1:]):
        raise SystemExit(f"成员 model_config 不一致，拒绝集成: {configs}")

    print("成员归一化（集成前必须确认是否同坐标系）：")
    for path, (_, meta) in zip(model_paths, members):
        print(f"  {path.name}: {json.dumps(meta['value_normalization'])}")

    nodes: list = []
    initializers: list = []
    terms: list[str] = []
    ref_meta = members[0][1]
    for index, (model, meta) in enumerate(members):
        prefix = f"e{index}_"
        member_nodes, member_inits, out_name = _prefixed(model, prefix)
        nodes += member_nodes
        initializers += member_inits
        mul, add = _affine(meta, ref_meta)
        initializers.append(numpy_helper.from_array(mul, f"{prefix}ens_mul_vec"))
        initializers.append(numpy_helper.from_array(add, f"{prefix}ens_add_vec"))
        nodes.append(helper.make_node("Mul", [out_name, f"{prefix}ens_mul_vec"], [f"{prefix}ens_scaled"], name=f"{prefix}ens_mul"))
        adj = f"{prefix}ens_adj"
        nodes.append(helper.make_node("Add", [f"{prefix}ens_scaled", f"{prefix}ens_add_vec"], [adj], name=f"{prefix}ens_add"))
        terms.append(adj)

    acc = terms[0]
    for step, term in enumerate(terms[1:]):
        nxt = f"ens_sum{step}"
        nodes.append(helper.make_node("Add", [acc, term], [nxt], name=nxt))
        acc = nxt
    initializers.append(numpy_helper.from_array(np.float32(len(terms)), "ens_count"))
    nodes.append(helper.make_node("Div", [acc, "ens_count"], ["output"], name="ens_div"))

    graph = helper.make_graph(
        nodes,
        "ramen_ensemble",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", INPUT_DIM])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", OUTPUT_DIM])],
        initializer=initializers,
    )
    proto = onnx.ModelProto()
    proto.CopyFrom(members[0][0])
    proto.graph.CopyFrom(graph)
    onnx.checker.check_model(proto)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(output_path))
    max_error = _verify(model_paths, members, output_path)

    sidecar = dict(ref_meta)
    sidecar["checkpoint"] = "ensemble"
    sidecar["onnx"] = str(output_path)
    sidecar["ensemble"] = {
        "mode": "affine-corrected elementwise mean",
        "members": [str(path) for path in model_paths],
        "reference_normalization": ref_meta["value_normalization"],
        "max_abs_error_vs_manual_mean": max_error,
        "dynamic_batch_tested": [1, 7],
        "member_count": len(model_paths),
    }
    _sidecar(output_path).write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"集成已写盘: {output_path}（+ {_sidecar(output_path).name}），成员数={len(model_paths)}，对拍误差={max_error:.3e}")
    return max_error


def _test_input(batch: int, seed: int) -> np.ndarray:
    """确定性随机输入（形状与 export_onnx.py 的保守测试输入同量级）。"""

    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.5, size=(batch, INPUT_DIM)).astype(np.float32)


def _verify(model_paths: list[Path], members, output_path: Path) -> float:
    """逐成员跑 onnxruntime 手工平均，与集成图输出对拍。"""

    import onnxruntime as ort

    ref = members[0][1]["value_normalization"]
    ref_center = np.asarray(ref["center"], dtype=np.float64)
    ref_scale = np.asarray(ref["scale"], dtype=np.float64)
    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in model_paths]
    ensemble = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for batch in (1, 7):
        x = _test_input(batch, 10_000 + batch)
        acc = np.zeros((batch, OUTPUT_DIM), dtype=np.float64)
        for session, (_, meta) in zip(sessions, members):
            out = session.run(["output"], {"input": x})[0].astype(np.float64)
            center = np.asarray(meta["value_normalization"]["center"], dtype=np.float64)
            scale = np.asarray(meta["value_normalization"]["scale"], dtype=np.float64)
            adj = out.copy()
            adj[:, VALUE_BASE:] = (out[:, VALUE_BASE:] * scale + center - ref_center) / ref_scale
            acc += adj
        expected = acc / len(sessions)
        got = ensemble.run(["output"], {"input": x})[0].astype(np.float64)
        error = float(np.max(np.abs(expected - got)))
        worst = max(worst, error)
        print(f"  batch={batch} 对拍 max_abs_err={error:.3e}")
    if worst >= TOLERANCE:
        raise SystemExit(f"集成对拍误差 {worst:.3e} >= {TOLERANCE}，拒绝交付")
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description="合成仿射校正平均的集成 ONNX")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compose(args.models, args.output)


if __name__ == "__main__":
    main()
