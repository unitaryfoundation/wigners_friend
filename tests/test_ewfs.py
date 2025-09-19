import pytest

from qiskit_aer import Aer
from qiskit_ibm_runtime.fake_provider import FakeFez

from ewfs.ewfs import EWFS


# --- Shared fixtures ---------------------------------------------------------


@pytest.fixture(scope="module")
def layout_and_flags():
    layout = [53, 52, 51, 50, 49, 48, 47, 46, 57, 67, 68, 69, 70, 71, 58]
    flags = [58]
    return layout, flags


@pytest.fixture(scope="module")
def backend_and_cmap():
    backend = Aer.get_backend("aer_simulator")
    cmap = FakeFez().coupling_map
    return backend, cmap


def _make_ewfs(alice_setting, bob_setting, backend, cmap, layout, flags, shots=10_000):
    return EWFS(
        alice_setting=alice_setting,
        bob_setting=bob_setting,
        backend=backend,
        shots=shots,
        coupling_map=cmap,
        layout=layout,
        strategy="majority_vote",
        flag_qubits=flags,
    )


# --- Helpers ----------------------------------------------------------------


def assert_prob_dict_close(actual_probs: dict[str, float], expected_probs: dict[str, float], tol: float = 0.03):
    """
    Assert two probability dictionaries are "close enough".
    - exact matching keys (to catch regressions in output support)
    - each probability within absolute tolerance tol
    - total probability approximately 1
    """
    # Keys must match exactly (order irrelevant)
    assert set(actual_probs.keys()) == set(expected_probs.keys()), (
        f"Keys mismatch.\nActual:   {sorted(actual_probs.keys())}\n" f"Expected: {sorted(expected_probs.keys())}"
    )

    # Each probability should be within tol of expected
    for k in expected_probs:
        a, e = actual_probs[k], expected_probs[k]
        assert abs(a - e) <= tol, f"Probability for key {k} differs: actual={a}, expected={e}, tol={tol}"

    # Sanity check normalization
    total = sum(actual_probs.values())
    assert abs(total - 1.0) <= 0.02, f"Total probability not ~1.0, got {total}"


def assert_edge_set_equal(actual_edges: list[tuple[int, int]], expected_edges: list[tuple[int, int]]):
    """
    Compare edges as sets (order can vary due to neighbor iteration order).
    """
    assert set(actual_edges) == set(expected_edges), (
        f"Edge set mismatch.\nActual:   {sorted(actual_edges)}\n" f"Expected: {sorted(expected_edges)}"
    )


# --- Tests: GHZ edges and Flag ops ------------------------------------------


def test_ghz_and_flag_ops(layout_and_flags, backend_and_cmap):
    layout, flags = layout_and_flags
    backend, cmap = backend_and_cmap

    ewfs = _make_ewfs("peek", "reverse_1", backend, cmap, layout, flags, shots=1000)

    # Force circuit initialization so ghz_ops and flag_ops are populated
    _ = ewfs.circuit  # triggers _initialize_circuit -> _ghz() and _flag_operations()

    # Expected edges exactly as printed in your run (device qubit labels, NOT indices)
    # NOTE: ewfs.ghz_ops stores TUPLES OF *DEVICE QUBITS* (as returned by BFS),
    # but we immediately map to indices when applying gates. We compare what was printed.
    expected_ghz_edges = [
        (52, 51),
        (51, 50),
        (50, 49),
        (49, 48),
        (48, 47),
        (47, 57),
        (47, 46),
        (57, 67),
        (67, 68),
        (68, 69),
        (69, 70),
        (70, 71),
    ]
    # In the class, ghz_ops are stored as device-ids, then mapped via layout_dict at use time.
    # The printouts you shared show these device-id pairs, so we compare as-is:
    assert_edge_set_equal(ewfs.ghz_ops, expected_ghz_edges)

    # Expected flag ops are printed as *indices in layout* (see ewfs._flag_operations)
    # Your logs show (2, 14) and (13, 14), order sometimes swapped → compare as a set.
    expected_flag_ops = {(2, 14), (13, 14)}
    assert set(ewfs.flag_ops) == expected_flag_ops, (
        f"Flag ops mismatch.\nActual:   {sorted(ewfs.flag_ops)}\n" f"Expected: {sorted(expected_flag_ops)}"
    )

    # Extra sanity: branch factor = charlie_size - 1
    expected_charlie_size = len(layout) - len(flags) - 2
    assert ewfs.charlie_size == expected_charlie_size
    assert ewfs.branch_factor == expected_charlie_size - 1


# --- Tests: Post-selected distributions -------------------------------------


@pytest.mark.parametrize(
    "alice_setting,bob_setting,expected_probs",
    [
        (
            "peek",
            "reverse_1",
            {
                "1000000000000": 0.4068,
                "0000000000000": 0.0902,
                "0111111111111": 0.4175,
                "1111111111111": 0.0855,
            },
        ),
        (
            "peek",
            "reverse_2",
            {
                "1111111111111": 0.4122,
                "0111111111111": 0.0958,
                "0000000000000": 0.4013,
                "1000000000000": 0.0907,
            },
        ),
        (
            "reverse_2",
            "reverse_1",
            {
                "10": 0.4451,
                "00": 0.0599,
                "01": 0.4353,
                "11": 0.0597,
            },
        ),
        (
            "reverse_2",
            "reverse_2",
            {
                "01": 0.4457,
                "00": 0.0536,
                "10": 0.4417,
                "11": 0.0590,
            },
        ),
    ],
)
def test_post_selected_distributions(layout_and_flags, backend_and_cmap, alice_setting, bob_setting, expected_probs):
    layout, flags = layout_and_flags
    backend, cmap = backend_and_cmap

    ewfs = _make_ewfs(alice_setting, bob_setting, backend, cmap, layout, flags, shots=10_000)

    # Run and post-select
    ps = ewfs.post_select_results
    assert (alice_setting, bob_setting) in ps, "Missing setting in post-selected results"
    actual_probs = ps[(alice_setting, bob_setting)]

    # Compare with a modest tolerance to allow shot noise & transpiler variance
    assert_prob_dict_close(actual_probs, expected_probs, tol=0.03)


# --- Tests: Decoded distribution for ("peek", "reverse_1") -------------------


def test_decoded_distribution_peek_reverse1(layout_and_flags, backend_and_cmap):
    layout, flags = layout_and_flags
    backend, cmap = backend_and_cmap

    ewfs = _make_ewfs("peek", "reverse_1", backend, cmap, layout, flags, shots=10_000)

    decoded = ewfs.decode_results(ewfs.post_select_results)

    # From your final "Decoding results..." block
    expected_decoded = {
        "10": 0.4129,
        "01": 0.4099,
        "11": 0.0887,
        "00": 0.0885,
    }

    assert ("peek", "reverse_1") in decoded
    actual = decoded[("peek", "reverse_1")]

    assert_prob_dict_close(actual, expected_decoded, tol=0.03)
