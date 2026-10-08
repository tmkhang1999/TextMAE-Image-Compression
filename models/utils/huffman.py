"""Huffman coding of the patch positions (side information). The code table is not counted in the bit rate."""
import heapq
from collections import Counter

import torch


def build_code(values):
    """{value: bit string} for a list of hashable values."""
    heap = [[freq, i, {v: ""}] for i, (v, freq) in enumerate(Counter(values).items())]
    heapq.heapify(heap)
    if len(heap) == 1:  # a single symbol still needs one bit
        return {next(iter(heap[0][2])): "0"}
    tie = len(heap)
    while len(heap) > 1:
        lo, hi = heapq.heappop(heap), heapq.heappop(heap)
        merged = {v: "0" + c for v, c in lo[2].items()}
        merged.update({v: "1" + c for v, c in hi[2].items()})
        heapq.heappush(heap, [lo[0] + hi[0], tie, merged])
        tie += 1
    return heap[0][2]


def huffman_encode(tensor):
    """Returns (bit string, code table)."""
    values = tensor.reshape(-1).tolist()
    table = build_code(values)
    return "".join(table[v] for v in values), table


def huffman_decode(bits, table, shape, device="cpu"):
    reverse, code, out = {c: v for v, c in table.items()}, "", []
    for bit in bits:
        code += bit
        if code in reverse:
            out.append(reverse[code])
            code = ""
    return torch.tensor(out, device=device).view(shape)
