import sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
src = sys.argv[1]; out = sys.argv[2]
ENC = "/Transpose_226_output_0"
def load(): return onnx.load(src)
def add_enc(m):
    m.graph.output.append(helper.make_tensor_value_info(ENC, TensorProto.FLOAT, [1, 12, 512]))
    return m
def add_head(m, hidden, name):
    g = m.graph; rng = np.random.default_rng(0)
    ws = []
    if hidden:
        w1 = rng.standard_normal((512, hidden)).astype(np.float32) * 0.04
        b1 = np.zeros(hidden, np.float32); w2 = rng.standard_normal((hidden, 1)).astype(np.float32) * 0.06
        ws = [("hw1", w1), ("hb1", b1), ("hw2", w2)]
        g.node.extend([helper.make_node("MatMul", [ENC, "hw1"], ["h1"]), helper.make_node("Add", ["h1", "hb1"], ["h1b"]),
                       helper.make_node("Relu", ["h1b"], ["h1r"]), helper.make_node("MatMul", ["h1r", "hw2"], ["slip_logit"])])
    else:
        ws = [("hw", rng.standard_normal((512, 1)).astype(np.float32) * 0.04)]
        g.node.append(helper.make_node("MatMul", [ENC, "hw"], ["slip_logit"]))
    for n, w in ws: g.initializer.append(numpy_helper.from_array(w, n))
    g.output.append(helper.make_tensor_value_info("slip_logit", TensorProto.FLOAT, [1, 12, 1]))
    return m
onnx.save(add_enc(load()), f"{out}/enc.onnx")
onnx.save(add_head(load(), 0, "lin"), f"{out}/head_lin.onnx")
onnx.save(add_head(load(), 256, "mlp"), f"{out}/head_mlp256.onnx")
