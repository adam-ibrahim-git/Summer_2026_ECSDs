#!/usr/bin/env python3
"""
explore_ecsd.py — Enumerate ECSDs for given moduli, collect and group statistics.

For a fixed set of moduli [d1, d2, ...], tries every combination of offsets
in a range, computes number of components and cyclic vertices for each, and
groups results by those properties.

Usage
-----
    python3 explore_ecsd.py           # runs built-in families
    from explore_ecsd import explore, report  # import for custom searches

Key functions
-------------
    explore(moduli, offset_range, exact_only) → list of result dicts
    report(results, title)                    → prints grouped output
    is_exact_covering(array)                  → True if valid exact covering system
"""

from itertools import product
from collections import defaultdict
from math import lcm
from functools import reduce
from ecsd import ecsdstuff, cyclesizes, compress_sizes


# ── helpers ──────────────────────────────────────────────────────────────────

def is_exact_covering(array: list[list[int]]) -> bool:
    """Return True if array forms a valid exact covering system.
    Checks that every integer n in range(lcm of moduli) is covered exactly once."""
    mods    = [abs(item[0]) for item in array]
    offsets = [item[1] % abs(item[0]) for item in array]
    L = reduce(lcm, mods)
    return all(
        sum(1 for m, a in zip(mods, offsets) if n % m == a) == 1
        for n in range(L)
    )


# ── core exploration ──────────────────────────────────────────────────────────

def explore(
    moduli: list[int],
    offset_range: range | None = None,
    exact_only: bool = False,
) -> list[dict]:
    """
    Try all offset combinations for the given moduli and collect ECSD stats.

    Parameters
    ----------
    moduli       : list of d values, e.g. [2, 4, 4] or [-2, -4, -4]
    offset_range : offsets tried for every position; defaults to
                   range(lcm(|d|)) which covers all distinct residue patterns
    exact_only   : skip arrays that are not valid exact covering systems

    Returns
    -------
    List of dicts, one per array tried:
        array           [[d1,a1], ...]
        components      int
        cyclic_vertices int
        cycle_sizes     sorted list[int]
        is_exact        bool
    """
    abs_mods = [abs(d) for d in moduli]
    if offset_range is None:
        L = reduce(lcm, abs_mods)
        offset_range = range(L)

    results = []
    for offsets in product(offset_range, repeat=len(moduli)):
        array = [[d, a] for d, a in zip(moduli, offsets)]
        exact = is_exact_covering(array)
        if exact_only and not exact:
            continue
        _, _, cycles, verts = ecsdstuff(array)
        results.append({
            'array':           array,
            'components':      len(cycles),
            'cyclic_vertices': len(verts),
            'cycle_sizes':     sorted(cyclesizes(cycles)),
            'is_exact':        exact,
        })

    return results


# ── reporting ─────────────────────────────────────────────────────────────────

def report(results: list[dict], title: str = "", show_all: bool = True) -> None:
    """
    Print component-grouped statistics.

    Parameters
    ----------
    results  : output of explore()
    title    : optional header string
    show_all : if False, only print the summary table (not every array)
    """
    sep = "═" * 68
    if title:
        print(f"\n{sep}\n  {title}\n{sep}")

    n_exact = sum(1 for r in results if r['is_exact'])
    print(f"  {len(results)} arrays explored  |  {n_exact} valid exact covering systems\n")

    by_comp: dict[int, list] = defaultdict(list)
    for r in results:
        by_comp[r['components']].append(r)

    if show_all:
        for n_comp in sorted(by_comp):
            group = sorted(
                by_comp[n_comp],
                key=lambda r: (r['cyclic_vertices'], r['cycle_sizes'], str(r['array'])),
            )
            label = f"{n_comp} component{'s' if n_comp != 1 else ''}"
            exact_str = lambda r: "  ✓ exact" if r['is_exact'] else ""
            print(f"  ── {label} ({len(group)}) " + "─" * max(0, 52 - len(label)))
            for r in group:
                print(f"    {r['array']}{exact_str(r)}"
                      f"   cyclic={r['cyclic_vertices']}"
                      f"   sizes={compress_sizes(r['cycle_sizes'])}")
            print()

    # summary table
    print("  ── Summary " + "─" * 56)
    print(f"  {'components':>12}  {'arrays':>6}  exact  cyclic-vertex range  cycle-size patterns")
    for n_comp in sorted(by_comp):
        group = by_comp[n_comp]
        cvs   = [r['cyclic_vertices'] for r in group]
        nexact = sum(1 for r in group if r['is_exact'])
        size_patterns = sorted({compress_sizes(r['cycle_sizes']) for r in group})
        print(f"  {n_comp:>12}  {len(group):>6}  {nexact:>5}  "
              f"{min(cvs):>4}–{max(cvs):<4}     {', '.join(size_patterns)}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Family 1: moduli [2, 3, 4] ───────────────────────────────────────────
    # Note: 1/2 + 1/3 + 1/4 = 13/12 ≠ 1, so no valid exact covering systems.
    # These are non-standard arrays but still produce well-defined graphs.
    print("Exploring moduli [2, 3, 4]  (offsets 0..3 each)...")
    r1 = explore([2, 3, 4], offset_range=range(4))
    report(r1, "ECSD families  —  moduli [2, 3, 4]")

    # ── Family 2: moduli [2, 4, 4] ───────────────────────────────────────────
    # 1/2 + 1/4 + 1/4 = 1  →  valid exact covering systems exist.
    print("Exploring moduli [2, 4, 4]  (offsets 0..3 each)...")
    r2 = explore([2, 4, 4], offset_range=range(4))
    report(r2, "ECSD families  —  moduli [2, 4, 4]")

    # ── Family 2 exact-only ───────────────────────────────────────────────────
    exact2 = [r for r in r2 if r['is_exact']]
    if exact2:
        print(f"Exact-covering-system subset of [2, 4, 4]  ({len(exact2)} arrays):")
        for r in sorted(exact2, key=lambda r: (r['components'], r['cyclic_vertices'])):
            print(f"  {r['array']}   components={r['components']}"
                  f"   cyclic={r['cyclic_vertices']}"
                  f"   sizes={compress_sizes(r['cycle_sizes'])}")
        print()

    # ── Family 3: negative-base variant [-2, -4, -4] ─────────────────────────
    print("Exploring moduli [-2, -4, -4]  (offsets 0..3 each)...")
    r3 = explore([-2, -4, -4], offset_range=range(4))
    report(r3, "ECSD families  —  moduli [-2, -4, -4]")

    # ── Wider search: [2, 4, 4] with larger offset range ─────────────────────
    print("Wide search: moduli [2, 4, 4], offsets 0..11  (covers all mod-12 classes)...")
    r4 = explore([2, 4, 4], offset_range=range(12))
    report(r4, "Wide search  —  moduli [2, 4, 4], offsets 0–11", show_all=False)
