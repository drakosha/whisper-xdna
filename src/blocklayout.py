#!/usr/bin/env python3
"""Host-side block layouts for aie::mmul<4,8,4>, shared by the design and tests.

Deriving these from AMD's template rather than guessing:

  A  (matmul_vectorized_4x4): `pA + z*size_A` indexes the ROW-block z, and the
     k-step advances by `rowA*size_A`. So block index = k_blk*rowA + row_blk,
     each block the 4x8 sub-tile in row-major. "Column-major tiled" means the
     row-block varies fastest.

  B with transpose_b=true (the K operand): the comment says "block (n=j, k=i) at
     i*colB+j. Sub-tile elements are [n_in, k_in]". So index = k_blk*colB+n_blk
     and the sub-tile is 4 rows of n by 8 of k -- i.e. K tiles EXACTLY like A,
     with its own row count. The kernel applies aie::transpose itself.

  B with transpose_b=false (the V operand): loaded as-is, the hardware
     mul_4x8_4x8T does the transpose; sub-tile is [k_in, n_in], 8 by 4, and the
     index runs k-major as well.

  C (the score block G): "Block [rb, cb] at rb*size_C + cb*rowA_C*size_C", so
     index = col_blk*rowA + row_blk with 4x4 sub-tiles.

Everything here is a pure permutation, verified against the device in
mmul_layout.py before any of it is trusted.
"""
import numpy as np

R, S, T = 4, 8, 4


def tile_a(x):
    """(M, K) -> flat, blocks of 4x8, index = k_blk*(M/4) + row_blk."""
    m, k = x.shape
    assert m % R == 0 and k % S == 0, x.shape
    b = x.reshape(m // R, R, k // S, S)          # [rb, r, kb, s]
    return np.ascontiguousarray(b.transpose(2, 0, 1, 3)).reshape(-1)


def untile_a(v, m, k):
    return (v.reshape(k // S, m // R, R, S).transpose(1, 2, 0, 3)
            .reshape(m, k))


def tile_b_nomaj(x):
    """(N, K) laid out like A: 4x8 sub-tiles, k-major block order.

    This is the K operand of Q@K^T -- the kernel transposes each sub-tile."""
    return tile_a(x)


def tile_b_kmaj(x):
    """(K, N) -> flat, blocks of 8x4 ([k_in, n_in]), index = k_blk*(N/4)+n_blk.

    This is the V operand: no software transpose, the multiply does it."""
    k, n = x.shape
    assert k % S == 0 and n % T == 0, x.shape
    b = x.reshape(k // S, S, n // T, T)          # [kb, s, nb, t]
    return np.ascontiguousarray(b.transpose(0, 2, 1, 3)).reshape(-1)


def tile_b_nmaj(x):
    """(K, N) -> flat, blocks of 8x4 ([k_in, n_in]), index = n_blk*(K/8)+k_blk.

    The V operand. matmul_vectorized_4x4 with transpose_b=false indexes B as
    `pB + j*colA*size_B` with j the N-block and colA the number of K-blocks --
    n-major, unlike the K operand which is k-major. Confirmed on device:
    k-major V passes the V=ones test (which any permutation passes) but fails
    K=0, where P is uniform and the answer must be the column mean of V.
    """
    k, n = x.shape
    assert k % S == 0 and n % T == 0, x.shape
    b = x.reshape(k // S, S, n // T, T)          # [kb, s, nb, t]
    return np.ascontiguousarray(b.transpose(2, 0, 1, 3)).reshape(-1)


def tile_c(x):
    """(M, N) -> flat, blocks of 4x4, index = col_blk*(M/4) + row_blk."""
    m, n = x.shape
    b = x.reshape(m // R, R, n // T, T)
    return np.ascontiguousarray(b.transpose(2, 0, 1, 3)).reshape(-1)


def untile_c(v, m, n):
    return (v.reshape(n // T, m // R, R, T).transpose(1, 2, 0, 3)
            .reshape(m, n))
