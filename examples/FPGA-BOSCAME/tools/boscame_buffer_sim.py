"""Execute buffer-form experimental BOSCAME IR after linalg-to-loops.

This is a functional model, not a board or a performance model. It shares only
the textual block parser with the legacy simulator. Descriptor offsets and
aligned pointers are separate, scalar SPM access is rejected, and SPM persists
across programs. CPU cache policy for ordinary temporaries is inserted by the
final LLVM lowering; user DDR accesses are recorded here, including every DMA.
"""

import math
import re
import struct

from boscame_sim import (
    SimError,
    _split_top_level,
    after_name,
    op_name,
    read_module,
    result_name,
    split_op_and_types,
)

WIDTHS = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "index": 8, "f32": 4, "f64": 8}
SSA = r"%[\w.\-]+(?:#\d+)?"


def type_shape(text):
    head = text[text.index("<") + 1 :].split(",")[0].split(">")[0]
    fields = head.split("x")
    return [None if d in ("?", "*") else int(d) for d in fields[:-1]], fields[
        -1
    ]


def strides_for(shape):
    stride = 1
    strides = []
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return strides


def wrap(value, bits):
    return (int(value) + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def rounded(value, element):
    if element == "f32":
        return struct.unpack("<f", struct.pack("<f", value))[0]
    if element.startswith("i") and element != "index" and element != "i1":
        return wrap(value, int(element[1:]))
    return value


def element_type(type_text):
    """The scalar element of a supported scalar or one-dimensional vector."""
    type_text = type_text.strip()
    if type_text.startswith("vector<"):
        shape, element = type_shape(type_text)
        if len(shape) != 1:
            raise SimError("ordinary vector arithmetic must be one-dimensional")
        return element
    return type_text


class Buffer:
    def __init__(
        self,
        data,
        element,
        shape,
        strides=None,
        base=0,
        offset=0,
        bank=None,
        name=None,
    ):
        self.data, self.element, self.shape = data, element, shape
        self.strides = strides or strides_for(shape)
        self.base, self.offset, self.bank, self.name = base, offset, bank, name
        self.width = WIDTHS[element]

    def address(self, indices):
        if len(indices) != len(self.shape) or any(
            i < 0 or i >= n for i, n in zip(indices, self.shape)
        ):
            raise SimError("index outside memref shape")
        address = (
            self.base
            + (self.offset + sum(i * s for i, s in zip(indices, self.strides)))
            * self.width
        )
        if address < 0 or address + self.width > len(self.data):
            raise SimError("address outside backing storage")
        return address

    def load(self, indices):
        address = self.address(indices)
        if self.element.startswith("f"):
            return struct.unpack_from(
                "<f" if self.width == 4 else "<d", self.data, address
            )[0]
        return int.from_bytes(
            self.data[address : address + self.width], "little", signed=True
        )

    def store(self, indices, value):
        address = self.address(indices)
        if self.element.startswith("f"):
            struct.pack_into(
                "<f" if self.width == 4 else "<d", self.data, address, value
            )
        else:
            self.data[address : address + self.width] = (
                int(value) % (1 << (8 * self.width))
            ).to_bytes(self.width, "little")


class Machine:
    def __init__(self, path, function):
        self.function = next(
            op
            for op in read_module(path)
            if op_name(op.text) == "func.func"
            and op.blocks
            and "@" + function + "(" in op.text
        )
        self.spm = [
            bytearray([0xA5]) * (128 * 1024),
            bytearray([0x5A]) * (128 * 1024),
        ]
        self.accesses = []
        self.values = {}
        self.pending = {}
        self.live_matrix = False

    def value(self, text):
        text = text.strip()
        if text.startswith("%"):
            return self.values[text]
        return int(text)

    def result(self, op, values):
        name = result_name(op.text)
        if not name:
            return
        if ":" in name:
            prefix, count = name.split(":")
            if int(count) != len(values):
                raise SimError("result arity mismatch")
            for i, value in enumerate(values):
                self.values[f"{prefix}#{i}"] = value
        else:
            self.values[name] = values[0]

    def block(self, ops):
        for op in ops:
            yielded = self.run(op)
            if yielded is not None:
                return yielded
        return None

    def run(self, op):
        name = op_name(op.text)
        full = after_name(op.text)
        body, types = split_op_and_types(full)
        operands = re.findall(SSA, body)
        get = self.value
        if name == "arith.constant":
            payload = body[len(name) :].strip()
            element = element_type(types)

            def scalar(text):
                return rounded(
                    float(text)
                    if element.startswith("f")
                    else int(text == "true")
                    if text in ("true", "false")
                    else int(text),
                    element,
                )

            if types.startswith("vector<"):
                lanes = type_shape(types)[0][0]
                dense = re.fullmatch(r"dense<(.*)>", payload)
                if not dense:
                    raise SimError("unsupported vector constant: " + op.text)
                entries = dense[1].strip()
                value = (
                    [
                        scalar(entry.strip())
                        for entry in entries[1:-1].split(",")
                    ]
                    if entries.startswith("[")
                    else [scalar(entries)] * lanes
                )
                if len(value) != lanes:
                    raise SimError("vector constant lane count mismatch")
            else:
                value = scalar(payload)
            self.result(op, [value])
        elif name.startswith("arith.") or name.startswith("math."):
            args = [get(v) for v in operands]
            code = name.split(".")[1]
            result_element = element_type(
                _split_top_level(types.split(" to ")[-1])[-1]
            )
            source_element = element_type(types.split(" to ")[0])

            def evaluate(values):
                unsigned_codes = {
                    "divui",
                    "remui",
                    "minui",
                    "maxui",
                    "extui",
                    "uitofp",
                }
                predicate = body.split()[1].strip(",") if code == "cmpi" else ""
                if code in unsigned_codes or predicate.startswith("u"):
                    bits = (
                        64
                        if source_element == "index"
                        else int(source_element[1:])
                    )
                    values = [int(value) % (1 << bits) for value in values]
                if code in ("addi", "addf"):
                    value = values[0] + values[1]
                elif code in ("subi", "subf"):
                    value = values[0] - values[1]
                elif code in ("muli", "mulf"):
                    value = values[0] * values[1]
                elif code == "divf":
                    value = values[0] / values[1]
                elif code in ("divui", "divsi"):
                    value = abs(values[0]) // abs(values[1])
                    if (values[0] < 0) != (values[1] < 0):
                        value = -value
                elif code == "remui":
                    value = values[0] % values[1]
                elif code in ("minsi", "minui"):
                    value = min(values)
                elif code in ("maxsi", "maxui"):
                    value = max(values)
                elif code == "cmpi":
                    a, b = values
                    value = {
                        "eq": a == b,
                        "ne": a != b,
                        "slt": a < b,
                        "ult": a < b,
                        "sle": a <= b,
                        "ule": a <= b,
                        "sgt": a > b,
                        "ugt": a > b,
                        "sge": a >= b,
                        "uge": a >= b,
                    }[predicate]
                elif code == "select":
                    value = values[1] if values[0] else values[2]
                elif code == "andi":
                    value = values[0] & values[1]
                elif code == "ori":
                    value = values[0] | values[1]
                elif code in (
                    "index_cast",
                    "index_castui",
                    "extsi",
                    "trunci",
                    "extf",
                    "truncf",
                    "sitofp",
                ) or code in ("extui", "uitofp"):
                    value = values[0]
                elif code == "negf":
                    value = -values[0]
                elif code == "rsqrt":
                    value = 1.0 / math.sqrt(values[0])
                elif code == "sqrt":
                    value = math.sqrt(values[0])
                else:
                    raise SimError("unsupported arithmetic: " + op.text)
                # Each emitted operation rounds independently, including each
                # lane. This does not borrow the test's independent oracle.
                return rounded(value, result_element)

            vector_lengths = {
                len(value) for value in args if isinstance(value, list)
            }
            if vector_lengths:
                if len(vector_lengths) != 1:
                    raise SimError("vector arithmetic lane count mismatch")
                value = [
                    evaluate(
                        [
                            arg[i] if isinstance(arg, list) else arg
                            for arg in args
                        ]
                    )
                    for i in range(next(iter(vector_lengths)))
                ]
            else:
                value = evaluate(args)
            self.result(op, [value])
        elif name == "vector.broadcast":
            shape, element = type_shape(types.split(" to ")[-1])
            if len(shape) != 1 or isinstance(get(operands[0]), list):
                raise SimError(
                    "only scalar-to-1D vector broadcast is supported"
                )
            self.result(op, [[rounded(get(operands[0]), element)] * shape[0]])
        elif name == "vector.reduction":
            if not re.match(r"vector.reduction\s+<add>,", body):
                raise SimError("only ordered ADD vector reduction is supported")
            if "reassoc" in body:
                raise SimError("reassociated vector reduction is unsupported")
            element = types.split(" into ")[-1].strip()
            values = get(operands[0])
            if not isinstance(values, list) or any(
                isinstance(v, list) for v in values
            ):
                raise SimError(
                    "vector reduction requires a one-dimensional vector"
                )
            if len(operands) == 2:
                value = get(operands[1])
            else:
                value, values = values[0], values[1:]
            for lane in values:
                value = rounded(value + lane, element)
            self.result(op, [value])
        elif name == "vector.extract":
            values = get(operands[0])
            positions = re.search(r"\[(.*?)\]", body)
            if positions is None:
                raise SimError("vector extract requires an explicit position")
            positions = _split_top_level(positions[1])
            if len(positions) != 1 or not isinstance(values, list):
                raise SimError(
                    "only scalar extraction from a 1D vector is supported"
                )
            position = get(positions[0])
            if position < 0 or position >= len(values):
                raise SimError("vector extract index outside vector")
            self.result(op, [values[position]])
        elif name in ("memref.alloc", "memref.alloca"):
            shape, element = type_shape(types)
            if any(n is None for n in shape):
                raise SimError(
                    "dynamic allocation is unsupported by this simulator"
                )
            self.result(
                op,
                [
                    Buffer(
                        bytearray(math.prod(shape) * WIDTHS[element]),
                        element,
                        shape,
                    )
                ],
            )
        elif name == "bosc_ame.spm.view":
            bank = int(re.search(r"bank = (\d+)", body)[1])
            offset = int(re.search(r"offset = (\d+)", body)[1])
            shape, element = type_shape(types)
            self.result(
                op,
                [
                    Buffer(
                        self.spm[bank], element, shape, base=offset, bank=bank
                    )
                ],
            )
        elif name in ("memref.cast", "memref.memory_space_cast"):
            self.result(op, [get(operands[0])])
        elif name == "memref.reinterpret_cast":
            source = get(operands[0])
            offset, sizes, strides = re.findall(r"\[(.*?)\]", full)[:3]
            shape, element = type_shape(full.rsplit(" to ", 1)[-1])
            self.result(
                op,
                [
                    Buffer(
                        source.data,
                        element,
                        [get(x) for x in _split_top_level(sizes)],
                        [get(x) for x in _split_top_level(strides)],
                        base=source.base,
                        offset=get(offset),
                        bank=source.bank,
                        name=source.name,
                    )
                ],
            )
        elif name == "memref.subview":
            source = get(operands[0])
            offsets, sizes, strides = re.findall(r"\[(.*?)\]", body)[:3]
            offsets, sizes, strides = (
                [get(x) for x in _split_top_level(values)]
                for values in (offsets, sizes, strides)
            )
            target_shape, _ = type_shape(types.split(" to ")[-1])
            target_strides = [a * b for a, b in zip(source.strides, strides)]
            offset = source.offset + sum(
                a * b for a, b in zip(offsets, source.strides)
            )
            while len(sizes) > len(target_shape):
                dimension = sizes.index(1)
                sizes.pop(dimension)
                target_strides.pop(dimension)
            self.result(
                op,
                [
                    Buffer(
                        source.data,
                        source.element,
                        sizes,
                        target_strides,
                        base=source.base,
                        offset=offset,
                        bank=source.bank,
                        name=source.name,
                    )
                ],
            )
        elif name == "memref.collapse_shape":
            source = get(operands[0])
            shape, _ = type_shape(types.split(" into ")[-1])
            self.result(
                op,
                [
                    Buffer(
                        source.data,
                        source.element,
                        shape,
                        base=source.base,
                        offset=source.offset,
                        bank=source.bank,
                        name=source.name,
                    )
                ],
            )
        elif name == "memref.extract_strided_metadata":
            source = get(operands[0])
            base = Buffer(
                source.data,
                source.element,
                [],
                base=source.base,
                bank=source.bank,
                name=source.name,
            )
            self.result(
                op, [base, source.offset, *source.shape, *source.strides]
            )
        elif name in (
            "memref.load",
            "memref.store",
            "vector.load",
            "vector.store",
        ):
            store = name.endswith("store")
            buffer = get(operands[1 if store else 0])
            if buffer.bank is not None and name.startswith("memref."):
                raise SimError("scalar SPM access")
            if self.pending:
                raise SimError("memory access before DMA wait")
            indices = [
                get(v)
                for v in _split_top_level(re.search(r"\[(.*?)\]", body)[1])
            ]
            if name.startswith("vector."):
                lanes = type_shape(types[types.index("vector<") :])[0]
                if len(lanes) != 1:
                    raise SimError(
                        "ordinary vector transfer must be one-dimensional"
                    )
                values = get(operands[0]) if store else []
                for i in range(lanes[0]):
                    idx = indices[:-1] + [indices[-1] + i]
                    if store:
                        buffer.store(idx, values[i])
                    else:
                        values.append(buffer.load(idx))
                if not store:
                    self.result(op, [values])
            elif store:
                buffer.store(indices, get(operands[0]))
            else:
                self.result(op, [buffer.load(indices)])
            if buffer.name is not None:
                address = buffer.address(indices)
                size = buffer.width * (
                    lanes[0] if name.startswith("vector.") else 1
                )
                self.accesses.append(
                    (
                        buffer.name,
                        address,
                        address + size,
                        "write" if store else "read",
                    )
                )
        elif name == "memref.copy":
            source, destination = (get(v) for v in operands[:2])
            if (
                source.bank is not None
                or destination.bank is not None
                or source.shape != destination.shape
            ):
                raise SimError("unsupported memref.copy, including SPM copy")
            import itertools

            for indices in itertools.product(*(range(n) for n in source.shape)):
                destination.store(indices, source.load(indices))
                for buffer, kind in ((source, "read"), (destination, "write")):
                    if buffer.name is not None:
                        address = buffer.address(indices)
                        self.accesses.append(
                            (buffer.name, address, address + buffer.width, kind)
                        )
        elif name == "scf.for":
            match = re.search(
                rf"scf.for ({SSA}) = ({SSA}) to ({SSA}) step ({SSA})(?: iter_args\((.*?)\))?",
                body,
            )
            carried = []
            for part in _split_top_level(match[5] or ""):
                key, initial = part.split("=")
                carried.append((key.strip(), get(initial)))
            for i in range(get(match[2]), get(match[3]), get(match[4])):
                self.values[match[1]] = i
                for key, value in carried:
                    self.values[key] = value
                values = self.block(op.blocks[0]) or []
                carried = [
                    (key, value) for (key, _), value in zip(carried, values)
                ]
            self.result(op, [value for _, value in carried])
        elif name == "scf.if":
            branch = 0 if get(operands[0]) else 1
            yielded = (
                self.block(op.blocks[branch]) if branch < len(op.blocks) else []
            )
            self.result(op, yielded or [])
        elif name in ("scf.yield", "func.return", "return"):
            return [get(v) for v in operands]
        elif name == "bosc_ame.dma.start":
            if self.live_matrix:
                raise SimError("DMA call inside matrix live range")
            channel = int(re.search(r"channel = (\d+)", body)[1])
            source, destination, length = (get(v) for v in operands[:3])
            src, dst = source.address([0]), destination.address([0])
            if (
                channel in self.pending
                or length <= 0
                or length % 64
                or src % 64
                or dst % 64
            ):
                raise SimError("invalid DMA contract")
            if (
                length > source.shape[0] * source.width
                or length > destination.shape[0] * destination.width
            ):
                raise SimError("DMA exceeds endpoint")
            if src + length > len(source.data) or dst + length > len(
                destination.data
            ):
                raise SimError("DMA exceeds backing storage")
            self.pending[channel] = (source, destination, src, dst, length)
            for buffer, address, kind in (
                (source, src, "read"),
                (destination, dst, "write"),
            ):
                if buffer.name is not None:
                    self.accesses.append(
                        (buffer.name, address, address + length, kind)
                    )
            self.result(op, [channel])
        elif name == "bosc_ame.dma.wait":
            source, destination, src, dst, length = self.pending.pop(
                get(operands[0])
            )
            destination.data[dst : dst + length] = source.data[
                src : src + length
            ]
        elif name.startswith("bosc_ame.tile.load_"):
            if self.pending:
                raise SimError("tile read before DMA wait")
            source = get(operands[0])
            shape, _ = type_shape(types.split(" -> ")[-1])
            self.result(
                op,
                [
                    [
                        [source.load([i, j]) for j in range(shape[1])]
                        for i in range(shape[0])
                    ]
                ],
            )
            self.live_matrix = True
        elif name == "bosc_ame.tile.mma":
            c, a, b = (get(v) for v in operands[:3])
            self.result(
                op,
                [
                    [
                        [
                            wrap(
                                c[i][j]
                                + sum(a[i][k] * b[k][j] for k in range(len(b))),
                                32,
                            )
                            for j in range(len(b[0]))
                        ]
                        for i in range(len(a))
                    ]
                ],
            )
        elif name == "bosc_ame.tile.store_acc":
            matrix, destination = get(operands[0]), get(operands[1])
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    destination.store([i, j], value)
            self.live_matrix = False
        elif name in ("func.call", "call"):
            if self.live_matrix:
                raise SimError("runtime call inside matrix live range")
            if not re.search(r"@bosc_ame_ddr_(read|write)_begin", body):
                raise SimError("unsupported runtime call: " + op.text)
        elif name in ("memref.dealloc",):
            pass
        else:
            raise SimError("unsupported operation: " + op.text)
        return None

    def launch(self, buffers, scalars=None):
        scalars = scalars or {}
        self.values = {}
        signature = self.function.text.split("(", 1)[1].split(")", 1)[0]
        for part in _split_top_level(signature):
            name, type_text = part.split(":", 1)
            name = name.strip()
            if "memref<" in type_text:
                shape, element = type_shape(type_text)
                data = buffers[name]
                shape = (
                    [len(data) // WIDTHS[element]] if None in shape else shape
                )
                self.values[name] = Buffer(data, element, shape, name=name)
            else:
                self.values[name] = scalars.get(name, 0)
        self.block(self.function.blocks[0])
        if self.pending or self.live_matrix:
            raise SimError("unfinished DMA or matrix values at return")
