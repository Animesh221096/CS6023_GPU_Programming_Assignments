#!/usr/bin/env python3
"""Convert between the binary and text formats of the A4 wildfire files.

  python3 convert.py in2bin  <input.txt> <input.bin>
  python3 convert.py in2txt  <input.bin> <input.txt>
  python3 convert.py out2txt <input.bin> <output.bin> <output.txt>

The text input format holds the same numbers as the binary format, in the
same order, separated by whitespace. Anything after a '#' on a line is a
comment. The output needs the input file too, to know N and the crew count.
"""
import struct
import sys


def read_input(path):
    """Parse a binary input file into a dict (grids as lists of rows)."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0

    def ints(k):
        nonlocal pos
        vals = struct.unpack_from("<%di" % k, data, pos)
        pos += 4 * k
        return list(vals)

    p = dict(zip("N T H dr dc W S D I R X".split(), ints(11)))
    N = p["N"]
    for name in ("fuel", "elev"):
        p[name] = [list(data[pos + r * N: pos + (r + 1) * N]) for r in range(N)]
        pos += N * N
    for name, width in (("fires", 2), ("stations", 2), ("crews", 3)):
        k = ints(1)[0]
        flat = ints(width * k)
        p[name] = [tuple(flat[width * i: width * (i + 1)]) for i in range(k)]
    if pos != len(data):
        raise ValueError("trailing bytes in input file")
    return p


def text_tokens(path):
    tokens = []
    with open(path) as f:
        for line in f:
            tokens += line.split("#", 1)[0].split()
    return [int(t) for t in tokens]


def in2bin(txt_path, bin_path):
    v = text_tokens(txt_path)
    pos = 0

    def take(k):
        nonlocal pos
        out = v[pos:pos + k]
        if len(out) != k:
            raise ValueError("text input ended early")
        pos += k
        return out

    out = bytearray()
    header = take(11)
    N = header[0]
    out += struct.pack("<11i", *header)
    for _ in range(2):                       # fuel, elevation
        cells = take(N * N)
        if any(not 0 <= x <= 255 for x in cells):
            raise ValueError("grid values must be 0..255")
        out += bytes(cells)
    for width in (2, 2, 3):                  # fires, stations, crews
        k = take(1)[0]
        out += struct.pack("<i", k)
        out += struct.pack("<%di" % (width * k), *take(width * k))
    if pos != len(v):
        raise ValueError("extra numbers at the end of the text input")
    with open(bin_path, "wb") as f:
        f.write(out)


def in2txt(bin_path, txt_path):
    p = read_input(bin_path)
    lines = [" ".join(str(p[k]) for k in "N T H dr dc W S D I R X".split())
             + "    # N T H dr dc W S D I R X"]
    lines.append("# fuel")
    lines += [" ".join(map(str, row)) for row in p["fuel"]]
    lines.append("# elevation")
    lines += [" ".join(map(str, row)) for row in p["elev"]]
    for name in ("fires", "stations", "crews"):
        items = p[name]
        lines.append("%d    # %s" % (len(items), name))
        lines += [" ".join(map(str, it)) for it in items]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def out2txt(in_bin, out_bin, txt_path):
    p = read_input(in_bin)
    N, C = p["N"], len(p["crews"])
    with open(out_bin, "rb") as f:
        data = f.read()
    if len(data) != 2 * N * N + 12 * C:
        raise ValueError("output size %d does not match N=%d, C=%d" % (len(data), N, C))
    lines = ["# state (0 unburnt, 1 burning, 2 burnt, 3 wet)"]
    lines += [" ".join(map(str, data[r * N:(r + 1) * N])) for r in range(N)]
    lines.append("# fuel")
    lines += [" ".join(map(str, data[N * N + r * N: N * N + (r + 1) * N])) for r in range(N)]
    lines.append("# crews (row col water)")
    for i in range(C):
        lines.append(" ".join(map(str, struct.unpack_from("<3i", data, 2 * N * N + 12 * i))))
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    cmds = {"in2bin": (in2bin, 2), "in2txt": (in2txt, 2), "out2txt": (out2txt, 3)}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds or len(sys.argv) != 2 + cmds[sys.argv[1]][1]:
        sys.exit(__doc__)
    fn, _ = cmds[sys.argv[1]]
    fn(*sys.argv[2:])


if __name__ == "__main__":
    main()
