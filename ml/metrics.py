"""Receiver-selection evaluation metrics.

The engineering objective is receiver-selection quality, not neural-network
loss: given the selected receiver per timestamp, how good is the resulting
position, how often the selector picks the better receiver, and how stable is
the switching behaviour.
"""
import numpy as np

from baselines import selected_errors


def apply_hysteresis_sessions(W, available, session, threshold):
    """Hysteresis selection applied independently within each session."""
    from predict import apply_hysteresis
    W = np.asarray(W)
    available = np.asarray(available, dtype=bool)
    session = np.asarray(session)
    decisions = np.full(len(W), -1, dtype=int)
    boundaries = np.r_[0, np.flatnonzero(session[1:] != session[:-1]) + 1, len(W)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        decisions[start:end] = apply_hysteresis(W[start:end], available[start:end], threshold)
    return decisions


def oracle_decisions(errors, available):
    """Best possible receiver (lowest actual error) among available ones."""
    errors = np.where(np.asarray(available, dtype=bool), np.asarray(errors, dtype=float), np.inf)
    sel = np.argmin(errors, axis=-1)
    return np.where(np.asarray(available).any(axis=-1), sel, -1)


def switching_stats(decisions, session, errors):
    """Switches, unnecessary switches and mean selection duration.

    A switch occurs when the selected receiver changes between consecutive
    timestamps inside one session (sessions are independent experiments).
    A switch is unnecessary if the newly selected receiver has an error >= the
    receiver that was previously selected (the switch did not improve the
    selected position at that instant). -1 (no selection) breaks a run.
    """
    decisions = np.asarray(decisions)
    session = np.asarray(session)
    errors = np.asarray(errors, dtype=float)
    switches = unnecessary = beneficial = 0
    runs = run_len = run_len_sum = 0
    prev = -1
    for i in range(len(decisions)):
        d = decisions[i]
        new_session = i == 0 or session[i] != session[i - 1]
        if d < 0:
            if run_len:
                runs += 1
                run_len_sum += run_len
                run_len = 0
            prev = -1
            continue
        if prev < 0:
            prev, run_len = d, 1
            continue
        if new_session:
            if run_len:
                runs += 1
                run_len_sum += run_len
            prev, run_len = d, 1
            continue
        if d != prev:
            if run_len:
                runs += 1
                run_len_sum += run_len
            switches += 1
            if errors[i, d] < errors[i, prev]:
                beneficial += 1
            else:
                unnecessary += 1
            prev, run_len = d, 1
        else:
            run_len += 1
    if run_len:
        runs += 1
        run_len_sum += run_len
    return dict(switches=int(switches), beneficial_switches=int(beneficial),
                unnecessary_switches=int(unnecessary),
                mean_duration_s=float(run_len_sum / runs) if runs else 0.0)


def evaluate_selection(decisions, errors, available, session):
    """Full receiver-selection metric block for one selector."""
    decisions = np.asarray(decisions)
    errors = np.asarray(errors, dtype=float)
    available = np.asarray(available, dtype=bool)
    available_epochs = int(available.any(axis=-1).sum())
    sel_err = selected_errors(decisions, errors, available)
    oracle = oracle_decisions(errors, available)

    ok = available.any(axis=-1)
    # selection accuracy: fraction of available epochs where selector == oracle
    m = (decisions[ok] == oracle[ok])
    acc = float(m.mean()) if ok.any() else 0.0

    stats = {
        'available_epochs': available_epochs,
        'selection_accuracy_vs_oracle': acc,
        'mean_error_m': float(sel_err.mean()) if len(sel_err) else None,
        'median_error_m': float(np.median(sel_err)) if len(sel_err) else None,
        'rmse_m': float(np.sqrt(np.mean(sel_err ** 2))) if len(sel_err) else None,
        'p95_error_m': float(np.percentile(sel_err, 95)) if len(sel_err) else None,
        'max_error_m': float(sel_err.max()) if len(sel_err) else None,
    }
    stats.update(switching_stats(decisions, session, errors))
    return stats