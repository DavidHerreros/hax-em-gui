"""Magic + version prelude for the viewers' hand-packed binary responses.

Every binary payload (meshes, grids, matrices, vectors) starts with an 8-byte
prelude: ``b"HXW1"`` + format id (u8) + format version (u8) + two reserved
zero bytes. Before this, a frontend/backend mismatch — or an HTML error page
fetched as ``arrayBuffer()`` — was parsed as geometry and rendered as garbage;
now the client refuses it with a message that names the problem. The client
side of this contract lives in ``static/wire.js``.
"""

from __future__ import annotations

MAGIC = b"HXW1"

MESH = 1        # int32 nverts,nfaces | f32 level,mass,vmax | verts | normals | faces
GRID = 2        # int32 nx,ny,nz | f32 vmin,vmax,mean,std | f32 boxLo[3],boxHi[3] | u8 data
F32_MATRIX = 3  # int32 rows,cols | f32 data (C order)
I32_VECTOR = 4  # int32 n | i32 data
F32_VECTOR = 5  # int32 n | f32 data


def header(fmt: int, version: int = 1) -> bytes:
    return MAGIC + bytes([fmt, version, 0, 0])
