#!/usr/bin/env python3
"""
enum_degree3.py — Enumerate all degree-3 exact covering system digraphs.

Valid degree-3 modulus families (|moduli| that can partition Z):
  - {3, 3, 3}  (1/3 + 1/3 + 1/3 = 1)
  - {2, 4, 4}  (1/2 + 1/4 + 1/4 = 1)

For each family, tries every sign combination of the base moduli and every
offset combination, deduplicates by canonical rule set, and reports the
1-component and max-component cases.
"""

from itertools import product
from collections import defaultdict
from math import lcm
from functools import reduce

from ecsd import ecsdstuff, cyclesizes, compress_sizes, is_exact_covering


def _canonical_key(array: list[list[int]]) -> frozenset:
    """Deduplication key: frozenset of (d, a mod |d|) — order-independent, offset-normalized."""
    return frozenset((d, a % abs(d)) for d, a in array)


def enum_degree3_family(base_moduli: tuple) -> list[dict]:
    """
    Enumerate all degree-3 ECSDs for one base modulus triple.

    Tries every sign combination of the moduli and every offset combination
    in range(lcm(|moduli|)), keeping only valid exact covering systems.
    Deduplicates by canonical rule set.
    """
    abs_mods = tuple(abs(d) for d in base_moduli)
    L = reduce(lcm, abs_mods)

    seen: set[frozenset] = set()
    results: list[dict] = []

    for signs in product([1, -1], repeat=3):
        moduli = tuple(s * m for s, m in zip(signs, abs_mods))

        for offsets in product(range(L), repeat=3):
            array = [[d, a] for d, a in zip(moduli, offsets)]

            if not is_exact_covering(array):
                continue

            key = _canonical_key(array)
            if key in seen:
                continue
            seen.add(key)

            _, _, cycles, verts = ecsdstuff(array)
            results.append({
                'array':           array,
                'components':      len(cycles),
                'cyclic_vertices': len(verts),
                'cycle_sizes':     sorted(cyclesizes(cycles)),
            })

    return results


def enum_all_degree3() -> list[dict]:
    """Enumerate all degree-3 ECSDs across both valid modulus families."""
    all_results = []

    print("Enumerating {3,3,3} family...")
    r1 = enum_degree3_family((3, 3, 3))
    print(f"  {len(r1)} distinct ECSDs")
    for r in r1:
        r['family'] = '3,3,3'
    all_results.extend(r1)

    print("Enumerating {2,4,4} family...")
    r2 = enum_degree3_family((2, 4, 4))
    print(f"  {len(r2)} distinct ECSDs")
    for r in r2:
        r['family'] = '2,4,4'
    all_results.extend(r2)

    return all_results


def _sign_label(array: list[list[int]]) -> str:
    signs = [item[0] > 0 for item in array]
    if all(signs):
        return 'all_pos'
    if not any(signs):
        return 'all_neg'
    return 'mixed'


def report_degree3(results: list[dict]) -> None:
    """Print the 1-component and max-component ECSDs, then note patterns."""
    sep = "═" * 72

    comp_counts = [r['components'] for r in results]
    max_comp = max(comp_counts)

    print(f"\n{sep}")
    print(f"  Degree-3 ECSD enumeration — {len(results)} distinct systems")
    print(sep)

    # Distribution table
    by_comp: dict[int, list] = defaultdict(list)
    for r in results:
        by_comp[r['components']].append(r)

    print("\n  ── Component distribution " + "─" * 46)
    print(f"  {'components':>12}  {'count':>6}  cycle-size patterns")
    for n_comp in sorted(by_comp):
        group = by_comp[n_comp]
        patterns = sorted({compress_sizes(r['cycle_sizes']) for r in group})
        print(f"  {n_comp:>12}  {len(group):>6}  {', '.join(patterns)}")

    # 1-component
    one_comp = sorted(
        by_comp.get(1, []),
        key=lambda r: (r['cyclic_vertices'], r['cycle_sizes'], str(r['array'])),
    )
    print(f"\n  ── 1-component ECSDs ({len(one_comp)}) " + "─" * 44)
    if one_comp:
        for r in one_comp:
            print(f"    {r['array']}"
                  f"   cyclic={r['cyclic_vertices']}"
                  f"   sizes={compress_sizes(r['cycle_sizes'])}"
                  f"   [{r['family']}]")
    else:
        print("    (none)")

    # Max-component
    max_list = sorted(
        by_comp.get(max_comp, []),
        key=lambda r: (r['cyclic_vertices'], r['cycle_sizes'], str(r['array'])),
    )
    print(f"\n  ── Max-component ECSDs  ({max_comp} components, {len(max_list)} systems) " + "─" * 20)
    for r in max_list:
        print(f"    {r['array']}"
              f"   cyclic={r['cyclic_vertices']}"
              f"   sizes={compress_sizes(r['cycle_sizes'])}"
              f"   [{r['family']}]")

    # Patterns
    print(f"\n  ── Patterns " + "─" * 60)

    for label, pred in [
        ('all-positive base', lambda r: all(item[0] > 0 for item in r['array'])),
        ('all-negative base', lambda r: all(item[0] < 0 for item in r['array'])),
        ('mixed-sign base',   lambda r: any(item[0] > 0 for item in r['array'])
                                     and any(item[0] < 0 for item in r['array'])),
    ]:
        group = [r for r in results if pred(r)]
        if group:
            comp_set = sorted({r['components'] for r in group})
            print(f"  {label:22s} ({len(group):3d} systems):  components ∈ {comp_set}")

    print()
    for label, pred in [
        ('balanced digits (Σaᵢ=0)',        lambda r: sum(item[1] for item in r['array']) == 0),
        ('non-negative digits (all aᵢ≥0)',  lambda r: all(item[1] >= 0 for item in r['array'])),
        ('standard {0,1,2} on base ±3',     lambda r: sorted(item[1] for item in r['array']) == [0,1,2]),
        ('standard {0,1,3} on base ±2,±4,±4', lambda r: sorted(item[1] for item in r['array']) == [0,1,3]),
    ]:
        group = [r for r in results if pred(r)]
        if group:
            comp_set = sorted({r['components'] for r in group})
            print(f"  {label:40s} ({len(group):3d}):  components ∈ {comp_set}")

    print()


if __name__ == "__main__":
    results = enum_all_degree3()
    report_degree3(results)
