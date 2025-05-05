import logging
from typing import TYPE_CHECKING, Optional

import mlop
from mlop.api import make_compat_graph_nodes_v1, make_compat_graph_v1
from mlop.util import import_lib

if TYPE_CHECKING:
    import torch
else:
    torch = import_lib("torch")

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Torch"


def _watch_torch(
    module: "torch.nn.Module",
    disable_graph: bool = True,
    disable_grad: bool = False,
    disable_param: bool = False,
    freq: Optional[int] = 1000,
    bins: Optional[int] = 64,
    op=None,
    **kwargs,
):
    # TODO: remove legacy compat
    if "log" in kwargs:
        disable_grad = kwargs["log"] not in ["gradients", "all"]
        disable_param = kwargs["log"] not in ["parameters", "all"]
    freq = kwargs.get("log_freq", freq)
    disable_graph = not kwargs.get("log_graph", not disable_graph)

    if mlop.ops is None or len(mlop.ops) == 0:
        logger.critical(f"{tag}: no runs to attach, please call mlop.init() first")
        return
    else:
        op = mlop.ops[-1] if not op else op
        op._torch, module._nodes = module, {}

    if not disable_grad and not hasattr(module, "_hook_grad"):
        module._hook_grad = []
        for name, param in module.named_parameters():
            if param.requires_grad and check_param(param, name):
                module._hook_grad.append(
                    param.register_hook(_backward(op, name, freq, bins))
                )

    if not disable_param and not hasattr(module, "_hook_param"):
        module._hook_param = [module.register_forward_hook(_forward(op, freq, bins))]

    if not hasattr(module, "_hook_graph"):
        module._hook_graph = []
        if not disable_graph:
            module._hook_graph.append(module.apply(_add_hooks))
        module._hook_graph.append(
            module.register_forward_hook(_forward_module(op, disable_graph))
        )


def _forward_module(op, disable_graph):
    c = [0]

    def f(module, inputs, outputs):
        if c[0] == 1:
            if op._torch._nodes:
                op._iface._post_v1(
                    op.settings.url_graph,
                    op._iface.headers,
                    make_compat_graph_v1(op.settings, "torch", op._torch._nodes),
                    client=op._iface.client_api,
                ) if op._iface else None
                op._torch._nodes = []
            return

        c[0] = 1
        # nodes = update_nodes(read_module(module).to_json(), _to_dict(module))
        ref = read_module(module).to_json() if not disable_graph else {}
        op._torch._nodes = make_compat_graph_nodes_v1(_to_dict(module), ref=ref)

    return f


def _to_dict(module):
    d = {
        "id": id(module),
        "type": module.__class__.__name__,
        **get_args(module),
        **get_params(module),
    }

    nodes = []
    for idx, (name, c) in enumerate(module._modules.items()):
        if c is None:
            continue
        n = {"name": name, "order": idx}
        n.update(_to_dict(c))
        nodes.append(n)

    if nodes:
        d["nodes"] = nodes

    return d


class ModuleInst:
    def __init__(self, module, inst_id, struct_id, num_in, num_out):
        self.module: torch.nn.Module = module
        self.inst_id = inst_id
        self.struct_id = struct_id
        self._graph = mlop.Graph()

        for i in range(num_in):
            self._graph.add_node(node=f"in_{i}", label=f"Input {i}")

        for i in range(num_out):
            self._graph.add_node(node=f"out_{i}", label=f"Output {i}")

    def _get_up_struct(self, module, inst_id):  # get upstream
        return self._up_filter(
            lambda node: node.module == module and node.inst_id == inst_id
        )

    def _up_filter(self, fn):
        for node in self._graph.nodes:
            if isinstance(node, ModuleInst):
                if fn(node):
                    return node
        return None

    def to_json(self):
        nodes = []
        for node in self._graph.nodes:
            if isinstance(node, str):
                nodes.append(
                    {
                        "label": self._graph.nodes[node].get("label"),
                        "node_id": str(node),
                        "node_type": "io",
                    }
                )
            else:
                nodes.append(
                    {
                        "label": f"{node.module.__class__.__name__} (Instance {node.inst_id})",
                        "node_id": f"s_{node.struct_id}",
                        "node_type": "module",
                        **node.to_json(),
                    }
                )

        edges = list(self._graph.edges())

        info = {
            "type": self.module.__class__.__name__,
            "id": id(self.module),
            "inst_id": str(self.inst_id),
            "nodes": nodes,
            "edges": [
                [
                    str(src) if isinstance(src, str) else f"s_{src.struct_id}",
                    str(dst) if isinstance(dst, str) else f"s_{dst.struct_id}",
                ]
                for src, dst in edges
            ],
        }

        return info


def read_module(module, inst_id=0):
    struct_id = [0]

    def process_grad_fn(  # use grad_fn to find upstream nodes
        struct: ModuleInst,
        grad_fn: torch.autograd.Function,
        i_grad_fn=None,
    ):
        if grad_fn is None:
            return []

        next_functions = grad_fn.next_functions
        if "BackwardHookFunctionBackward" in str(grad_fn) and i_grad_fn is not None:
            next_functions = (next_functions[i_grad_fn],)

        metadata = grad_fn.metadata

        # input nodes
        if "i_in" in metadata:
            assert metadata["module"] == struct.module
            assert metadata["inst_id"] == struct.inst_id
            return [
                {
                    "node": f"in_{metadata['i_in']}",
                    "index": metadata["i_in"],
                    "grad_fn": None,
                    "is_up": True,
                }
            ]

        # output nodes
        if "i_out" in metadata:
            up_module = metadata["module"]
            inst_id = metadata["inst_id"]

            up_struct = struct._get_up_struct(up_module, inst_id)

            if up_struct is None:
                up_struct = read_up(up_module, inst_id)
                struct._graph.add_node(node=up_struct, struct_id=up_struct.struct_id)

            return [
                {
                    "node": up_struct,
                    "index": metadata["i_out"],
                    "grad_fn": None,
                    "is_up": False,
                }
            ]

        # intermediate nodes using cache
        if "up_cache" in metadata and i_grad_fn in metadata["up_cache"]:
            return metadata["up_cache"][i_grad_fn]

        # intermediate nodes
        ups = [process_grad_fn(struct, n[0], n[1]) for n in next_functions]
        for i, _ in enumerate(ups):
            while i < len(ups) and isinstance(ups[i], (list, tuple)):
                ups[i : i + 1] = ups[i]

        if "up_cache" not in metadata:
            metadata["up_cache"] = {}
        metadata["up_cache"][i_grad_fn] = ups

        return ups

    def read_up(module, inst_id):
        structure = ModuleInst(
            module,
            inst_id,
            struct_id[0],
            len(module._metadata["grad_fn_in"][inst_id]),
            len(module._metadata["grad_fn_out"][inst_id]),
        )
        struct_id[0] += 1

        # start with output nodes going back
        to_process = []
        for i, output_grad_fn in enumerate(
            structure.module._metadata["grad_fn_out"][structure.inst_id]
        ):
            to_process.append(
                {
                    "node": f"out_{i}",
                    "index": i,
                    "grad_fn": output_grad_fn,
                    "is_up": False,
                }
            ) if output_grad_fn is not None else None

        # process nodes
        s = set()
        while to_process:
            down = to_process.pop(0)

            # find upstream nodes
            ups = process_grad_fn(structure, down["grad_fn"])

            for up in ups:
                structure._graph.add_edge(
                    up["node"],
                    down["node"],
                    up_i_out=up["index"],
                    down_i_in=down["index"],
                )

                if isinstance(up["node"], str):  # skip input
                    if up["node"].startswith("in_"):
                        continue

                if isinstance(up["node"], ModuleInst):
                    id_module = (
                        id(up["node"].module),
                        up["node"].inst_id,
                    )
                    if id_module not in s:
                        s.add(id_module)
                        for j, input_grad_fn in enumerate(
                            up["node"].module._metadata["grad_fn_in"][
                                up["node"].inst_id
                            ]
                        ):
                            to_process.append(
                                {
                                    "node": up["node"],
                                    "index": j,
                                    "grad_fn": input_grad_fn,
                                    "is_up": False,
                                }
                            ) if input_grad_fn is not None else None

        return structure

    return read_up(module, inst_id)


def _add_hooks(module):
    hooks = []

    def _pre_hook(module, inputs):
        inputs = _enforce_grad(inputs)
        module._metadata["grad_fn_in"][module._metadata["num_fwd"]] = _next_grad_fns(
            inputs
        )
        module._metadata["num_fwd"] += 1
        return _enforce_grad(inputs)

    def _post_hook(module, inputs, outputs):
        outputs, s = _enforce_tuple(outputs)

        _add_metadata(module, inputs, outputs)
        inst_id = module._metadata["num_fwd"] - 1
        assert inst_id >= 0
        outputs = _enforce_grad(outputs)
        module._metadata["grad_fn_out"][inst_id] = _grad_fns(outputs)
        outputs = _enforce_grad(outputs)

        for io, i_name in [
            (_next_grad_fns(inputs), "i_in"),
            (_grad_fns(outputs), "i_out"),
        ]:
            for i, grad_fn in enumerate(io):
                if grad_fn is not None:
                    assert isinstance(grad_fn.metadata, dict)

                    grad_fn.metadata["module"] = module
                    grad_fn.metadata["inst_id"] = inst_id
                    grad_fn.metadata[i_name] = i

        return outputs[0] if s else outputs

    module._metadata = {
        "grad_fn_in": {},
        "grad_fn_out": {},
        "num_fwd": 0,
    }
    if not module._metadata.get("tracking"):
        hooks.append(module.register_forward_pre_hook(_pre_hook))
        hooks.append(module.register_forward_hook(_post_hook))
        module._metadata["tracking"] = True

    return hooks


def _add_metadata(module, inputs, outputs):
    if hasattr(module, "_metadata"):
        module._metadata["shape"] = {
            "in": [get_shape(i) for i in inputs],
            "out": [get_shape(o) for o in outputs],
        }
        module._metadata["id"] = {
            "in": [t.data_ptr() if hasattr(t, "data_ptr") else id(t) for t in inputs],
            "out": [t.data_ptr() if hasattr(t, "data_ptr") else id(t) for t in outputs],
        }


def _enforce_tuple(i, t=torch.Tensor if torch is not None else None):
    s = i is None or isinstance(i, t)
    r = (i,) if s else i
    return r, s


def _enforce_grad(tensors):
    def process_tensor(t):
        d = torch.tensor(0.0, requires_grad=True)
        if t is None:
            return d
        return t + d if torch.is_floating_point(t) else t

    return tuple(process_tensor(t) for t in tensors)


def _grad_fns(tensors):
    def process_tensor(t):
        return t.grad_fn if t.requires_grad else None

    return tuple(process_tensor(t) for t in tensors)


def _next_grad_fns(tensors):
    # workaround from torchexplorer
    if tensors[0] is not None and "BackwardHookFunctionBackward" in str(
        tensors[0].grad_fn
    ):
        return tuple(f[0] for f in tensors[0].grad_fn.next_functions)

    return _grad_fns(tensors)


def get_params(module):
    # TODO: find a robust solution to get params efficiently
    info = {
        "params": {
            name: list(param.size())
            for name, param in module.named_parameters()
            if "." not in name
        }
    }
    return info if info["params"] else {}


def get_args(module):
    info = {
        "args": [],
        "kwargs": {},
    }

    if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        info["args"] = [int(module.in_channels), int(module.out_channels)]
    elif hasattr(module, "in_features") and hasattr(module, "out_features"):
        info["args"] = [int(module.in_features), int(module.out_features)]

    for k, v in module.__dict__.items():  # dir(module)
        if not k.startswith("_") and not callable(v):  # skip private attrs and methods
            if isinstance(v, torch.Size):
                v = tuple(v)
            elif hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    continue
            if (
                v is not None
                and v != ()
                and v != []
                and isinstance(v, (int, float, str, bool))
            ):
                info["kwargs"][k] = v

    return info


def get_shape(tensor, r=set()):
    if hasattr(tensor, "size"):
        return list(tensor.size())  # pytorch
    elif hasattr(tensor, "get_shape"):
        return tensor.get_shape().as_list()  # tensorflow
    elif hasattr(tensor, "shape"):
        return tensor.shape

    try:
        r.add(id(tensor))
        return [get_shape(i, r) if id(i) not in r else 0 for i in tensor]
    except TypeError:
        logger.debug(f"{tag}: tensor {tensor} is not iterable")
        return []


def _backward(op, name, freq, bins):
    c = [0]

    def f(grad):
        c[0] += 1
        if c[0] < freq:
            return
        c[0] = 0
        hist = make_compat_histogram_tensor(grad.data, bins)
        if hist is not None:
            op.log({f"{op.settings.x_grad_label}/{name}": hist}, step=op._step)

    return f


def _forward(op, freq, bins):
    c = [0]

    def f(module, input, output):
        c[0] += 1
        if c[0] < freq:
            return
        c[0] = 0

        for name, param in module.named_parameters():
            if check_param(param, name):
                hist = make_compat_histogram_tensor(param.data, bins)
                if hist is not None:
                    op.log(
                        {f"{op.settings.x_param_label}/{name}": hist}, step=op._step
                    )
                else:
                    logger.error(f"{tag}: {name} does not contain a valid tensor")

    return f


def check_param(param, name):
    if isinstance(param, torch.autograd.Variable):
        return True
    else:
        logger.error(
            f"{tag}: {name} is of type {type(param).__module__}.{type(param).__name__} and not a torch.Variable"
        )
        return False


def make_compat_histogram_tensor(tensor, bins=64):
    if isinstance(tensor, (tuple, list)):
        tensor = torch.cat([t.detach().clone().reshape(-1) for t in tensor])
    tensor = tensor.detach().clone()

    # handle sparse tensor zeros
    zeros = None
    if tensor.is_sparse:
        tensor = tensor.cpu().coalesce()
        values = tensor._values()
        zeros = tensor.numel() - values.numel()
        tensor = values

    flat = tensor.reshape(-1)
    if flat.is_cuda:
        try:
            flat.histc(bins=64)  # check for histc support
            if not isinstance(flat, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
                flat = flat.type(torch.cuda.FloatTensor)
        except RuntimeError:
            flat = flat.cpu()
    if not flat.is_cuda and not isinstance(
        flat, (torch.FloatTensor, torch.DoubleTensor)
    ):
        flat = flat.type(torch.FloatTensor)

    flat = make_compat_tensor(flat)
    if flat is None:
        return None

    # find histogram bounds
    tmin, tmax = flat.min().item(), flat.max().item()
    if zeros:
        tmin = min(0, tmin)
        tmax = max(0, tmax)
    if tmin > tmax:
        tmin, tmax = tmax, tmin

    if True:  # tmin != tmax:
        tensor = flat.histc(bins=bins, min=tmin, max=tmax)
        bins = torch.linspace(tmin, tmax, steps=bins + 1)
    else:  # use single bin if all values are the same
        tensor = torch.Tensor([flat.numel()])
        bins = torch.Tensor([tmin, tmax])

    # add back zeros from sparse tensor
    if zeros:
        mask = (bins[:-1] <= 0) & (bins[1:] > 0)
        if not mask.any():
            mask = torch.zeros_like(bins[:-1], dtype=torch.bool)
            mask[-1] = bins[-1] == 0
        tensor[mask] += zeros

    return mlop.Histogram(data=(tensor.tolist(), bins.tolist()), bins=None)


def make_compat_tensor(tensor):
    if tensor.shape == torch.Size([0]) or (~torch.isfinite(tensor)).all().item():
        return None  # invalid if empty or all inf/nan
    elif not torch.isfinite(tensor).all():
        return tensor[torch.isfinite(tensor)]  # remove inf/nan
    else:
        return tensor
