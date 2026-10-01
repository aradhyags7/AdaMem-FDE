"""
Unit regression tests for Embedded Memory Error Controller (Phase III).

Tests:
1. Controller startup grace period (prevents false expansion at t ~ 0 when M_K = 0).
2. Controller cooldown hysteresis (prevents rapid mode hunting/chattering).
3. Physical state norm normalization (prevents division by zero when M_K = 0).
"""

import pytest
import torch
from core.controllers.embedded_controller import MemoryErrorController


def test_controller_startup_grace_period_no_false_expansion():
    """
    Regression test: at t ~ 0 when auxiliary memory M_K = 0, the controller
    must NOT immediately trigger false mode expansion.
    """
    controller = MemoryErrorController(
        beta=0.85,
        delta_t=0.01,
        T=5.0,
        tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=32,
        delta_K=4,
    )
    # Zero initial auxiliary state M_K(0) = 0
    m_state = torch.zeros((8,), dtype=torch.float64)
    z_curr = torch.tensor([1.0, 0.0], dtype=torch.float64)

    # First 3 steps (step_idx = 0, 1, 2) must remain at K_init = 8
    for step in range(3):
        t = step * 0.01
        m_state, event = controller.step_adaptation(step, t, m_state, z_curr)
        assert event is None, f"Premature expansion triggered at startup step {step}!"
        assert controller.current_K == 8, f"Active modes changed during grace period at step {step}!"


def test_controller_cooldown_hysteresis():
    """
    Regression test: after a mode expansion occurs, the controller must enforce
    a cooldown period of 6 steps before another expansion can be triggered.
    """
    controller = MemoryErrorController(
        beta=0.85,
        delta_t=0.01,
        T=5.0,
        tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=32,
        delta_K=4,
    )
    # Large non-zero state that creates significant error exceeding tolerance
    m_state = torch.ones((8,), dtype=torch.float64) * 5.0
    z_curr = torch.tensor([0.1, 0.1], dtype=torch.float64)

    # Step 0, 1, 2: grace period
    for step in range(3):
        m_state, event = controller.step_adaptation(step, step * 0.01, m_state, z_curr)

    # Step 3: past grace period, error exceeds tolerance -> triggers expansion
    m_state, event1 = controller.step_adaptation(3, 0.03, m_state, z_curr)
    assert event1 is not None, "Expected expansion at step 3 when error > tol!"
    assert event1.event_type == "expand"
    assert event1.K_new == 12
    assert controller.current_K == 12
    assert controller._cooldown == 6, f"Expected cooldown = 6, got {controller._cooldown}"

    # Next steps during cooldown (steps 4 through 9): must NOT trigger further expansions
    for step in range(4, 10):
        m_state, event_cooldown = controller.step_adaptation(step, step * 0.01, m_state, z_curr)
        assert event_cooldown is None, f"Expansion triggered during cooldown at step {step}!"
        assert controller.current_K == 12


def test_controller_physical_state_normalization_no_nan():
    """
    Regression test: verify error estimator with zero physical state and zero memory state
    does not produce NaN or ZeroDivisionError.
    """
    controller = MemoryErrorController(
        beta=0.85,
        delta_t=0.01,
        T=5.0,
        tol=1e-3,
        K_init=8,
        K_min=4,
        K_max=32,
    )
    m_zero = torch.zeros((8,), dtype=torch.float64)
    z_zero = torch.zeros((2,), dtype=torch.float64)

    err = controller.estimate_error(m_zero, z_zero)
    assert not torch.isnan(torch.tensor(err)), "Error estimate is NaN!"
    assert not torch.isinf(torch.tensor(err)), "Error estimate is Inf!"
    assert err >= 0.0, f"Error estimate must be non-negative, got {err}"
