"""
Synthetic Multimodal Physiological Dataset Generator (v2)
=========================================================
Generates EDA, ECG, and Facial AU data for 12 players in a 15-minute
simulated horror game session with two jump scares and a tension ramp.

Main v2 changes:
  - HR responses now use delayed rise + slow recovery instead of immediate exponential decay.
  - The tension HR component decays after the second scare rather than ending abruptly.
  - SCR responses use a Bateman-like response with latency and longer recovery tails.
  - EDA tonic tension elevation decays gradually after the second scare.
  - Sampling uses np.arange(n) / sampling_rate to avoid small timestamp drift.

Events (with per-player time variation):
  - Jump Scare 1: ~3-5 min mark
  - Tension Ramp: starts ~6 min mark
  - Jump Scare 2: ~12 min mark
"""

import json
import os

import numpy as np
import pandas as pd


def _alpha_response(time_arr, event_time, amplitude, delay=0.8, rise_tau=2.0, decay_tau=45.0):
    """Delayed rise + exponential recovery, normalized so amplitude is the peak height.

    Useful for heart-rate responses where the peak should occur a few seconds after
    the event, followed by a recovery period that can last tens of seconds or more.
    """
    x = time_arr - event_time - delay
    y = np.zeros_like(time_arr, dtype=float)
    mask = x > 0
    if not np.any(mask):
        return y

    raw = (1.0 - np.exp(-x[mask] / rise_tau)) * np.exp(-x[mask] / decay_tau)
    max_raw = raw.max() if raw.size else 1.0
    if max_raw > 0:
        raw = raw / max_raw
    y[mask] = amplitude * raw
    return y


def _bateman_scr(time_arr, event_time, amplitude, latency=1.5, rise_tau=1.2, decay_tau=12.0):
    """Bateman-like skin conductance response.

    The response begins after a latency, rises over a few seconds, and has a
    longer recovery tail. It is normalized so amplitude is the peak height.
    """
    x = time_arr - event_time - latency
    y = np.zeros_like(time_arr, dtype=float)
    mask = x > 0
    if not np.any(mask):
        return y

    # Difference of exponentials: quick rise, slower decay.
    raw = np.exp(-x[mask] / decay_tau) - np.exp(-x[mask] / rise_tau)
    raw = np.maximum(raw, 0)
    max_raw = raw.max() if raw.size else 1.0
    if max_raw > 0:
        raw = raw / max_raw
    y[mask] = amplitude * raw
    return y


def _smooth_ramp_with_recovery(time_arr, start, peak_time, peak_value, rise_tau=90.0, recovery_tau=140.0):
    """Component that rises during tension and then recovers slowly after peak_time."""
    comp = np.zeros_like(time_arr, dtype=float)

    during = (time_arr >= start) & (time_arr < peak_time)
    comp[during] = peak_value * (1.0 - np.exp(-(time_arr[during] - start) / rise_tau))

    value_at_peak = peak_value * (1.0 - np.exp(-(peak_time - start) / rise_tau))
    after = time_arr >= peak_time
    comp[after] = value_at_peak * np.exp(-(time_arr[after] - peak_time) / recovery_tau)
    return comp


def generate_dataset(output_dir="synthetic_data", n_players=12, seed=42):
    """Generate the full synthetic dataset."""
    rng = np.random.default_rng(seed)

    DURATION = 15 * 60  # 15 minutes in seconds
    EDA_SR = 4          # Hz
    ECG_SR = 100        # Hz
    AU_SR = 30          # Hz

    # Per-player event timing (with variation)
    player_events = {}
    for p in range(n_players):
        js1 = rng.uniform(3 * 60, 5 * 60)              # Jump scare 1: 3-5 min
        tension_start = rng.uniform(5.5 * 60, 6.5 * 60)  # Tension: ~6 min
        js2 = rng.uniform(11.5 * 60, 12.5 * 60)       # Jump scare 2: ~12 min
        player_events[p] = {
            "jump_scare_1": js1,
            "tension_start": tension_start,
            "jump_scare_2": js2,
        }

    os.makedirs(output_dir, exist_ok=True)

    for p in range(n_players):
        player_id = f"player_{p+1:03d}"
        player_dir = os.path.join(output_dir, player_id)
        os.makedirs(player_dir, exist_ok=True)

        events = player_events[p]

        # Shared latent player traits.
        # These help make cross-player responses meaningfully different.
        reactivity = rng.uniform(0.75, 1.35)
        recovery_slowdown = rng.uniform(0.8, 1.7)
        motion_proneness = rng.uniform(0.7, 1.5)

        # =============================================
        # Generate EDA (4 Hz)
        # =============================================
        eda_samples = DURATION * EDA_SR
        eda_time = np.arange(eda_samples) / EDA_SR

        # Tonic component: slow drift (baseline ~2-5 µS, person-dependent)
        baseline = rng.uniform(2.0, 5.0)
        tonic = baseline + 0.25 * np.sin(2 * np.pi * eda_time / rng.uniform(240, 360))
        tonic += np.cumsum(rng.normal(0, 0.0008, eda_samples))

        # Gradual tonic elevation during tension, followed by slow recovery.
        tonic += _smooth_ramp_with_recovery(
            eda_time,
            start=events["tension_start"],
            peak_time=events["jump_scare_2"],
            peak_value=rng.uniform(0.55, 1.10) * reactivity,
            rise_tau=rng.uniform(70, 130),
            recovery_tau=rng.uniform(90, 220) * recovery_slowdown,
        )

        # Phasic component: SCR peaks at events and spontaneous responses.
        phasic = np.zeros(eda_samples)

        # Jump scare 1: clear SCR plus anticipatory smaller responses.
        phasic += _bateman_scr(
            eda_time,
            events["jump_scare_1"],
            amplitude=rng.uniform(1.2, 2.6) * reactivity,
            latency=rng.uniform(1.0, 2.5),
            rise_tau=rng.uniform(0.9, 1.8),
            decay_tau=rng.uniform(8, 20) * recovery_slowdown,
        )
        for offset in rng.uniform(-35, -6, rng.integers(1, 4)):
            phasic += _bateman_scr(
                eda_time,
                events["jump_scare_1"] + offset,
                amplitude=rng.uniform(0.25, 0.75) * reactivity,
                latency=rng.uniform(1.0, 2.5),
                rise_tau=rng.uniform(1.0, 2.0),
                decay_tau=rng.uniform(6, 15) * recovery_slowdown,
            )

        # Tension ramp: frequent small SCRs, not just a single continuous slope.
        tension_duration = events["jump_scare_2"] - events["tension_start"]
        n_tension_scrs = rng.integers(8, 15)
        for _ in range(n_tension_scrs):
            t_scr = events["tension_start"] + rng.uniform(0, tension_duration)
            phasic += _bateman_scr(
                eda_time,
                t_scr,
                amplitude=rng.uniform(0.15, 0.85) * reactivity,
                latency=rng.uniform(1.0, 3.0),
                rise_tau=rng.uniform(1.0, 2.2),
                decay_tau=rng.uniform(7, 18) * recovery_slowdown,
            )

        # Jump scare 2: stronger SCR, but recovery is still gradual.
        phasic += _bateman_scr(
            eda_time,
            events["jump_scare_2"],
            amplitude=rng.uniform(1.8, 3.8) * reactivity,
            latency=rng.uniform(1.0, 2.5),
            rise_tau=rng.uniform(0.9, 1.8),
            decay_tau=rng.uniform(12, 30) * recovery_slowdown,
        )

        # Random spontaneous SCRs throughout.
        n_spontaneous = rng.integers(5, 12)
        for _ in range(n_spontaneous):
            t_spont = rng.uniform(30, DURATION - 30)
            phasic += _bateman_scr(
                eda_time,
                t_spont,
                amplitude=rng.uniform(0.1, 0.45) * reactivity,
                latency=rng.uniform(1.0, 3.0),
                rise_tau=rng.uniform(1.0, 2.5),
                decay_tau=rng.uniform(5, 15) * recovery_slowdown,
            )

        # Combine + sensor noise + motion artefacts.
        eda_signal = tonic + phasic
        eda_noise = rng.normal(0, 0.06, eda_samples)

        # Single-sample spikes and short burst artefacts.
        n_spikes = rng.integers(3, 8)
        for _ in range(n_spikes):
            spike_idx = rng.integers(0, eda_samples)
            eda_noise[spike_idx] += rng.choice([-1, 1]) * rng.uniform(1.5, 4.5) * motion_proneness

        n_bursts = rng.integers(1, 4)
        for _ in range(n_bursts):
            burst_start = rng.integers(0, eda_samples - 20)
            burst_len = rng.integers(8, 35)  # 2-9 seconds at 4 Hz
            eda_noise[burst_start:burst_start + burst_len] += rng.normal(
                0, 0.18 * motion_proneness, burst_len
            )

        eda_signal += eda_noise
        eda_signal = np.clip(eda_signal, 0.01, None)

        eda_df = pd.DataFrame({
            "timestamp": eda_time,
            "eda_raw": eda_signal,
        })
        eda_df.to_csv(os.path.join(player_dir, "eda.csv"), index=False)

        # =============================================
        # Generate ECG (100 Hz)
        # =============================================
        ecg_samples = DURATION * ECG_SR
        ecg_time = np.arange(ecg_samples) / ECG_SR

        # Heart rate profile (bpm) - varies with events.
        base_hr = rng.uniform(65, 80)
        hr_profile = np.ones(ecg_samples) * base_hr

        # Tension ramp: gradual increase, with slow recovery after the second scare.
        tension_peak = rng.uniform(8, 16) * reactivity
        hr_profile += _smooth_ramp_with_recovery(
            ecg_time,
            start=events["tension_start"],
            peak_time=events["jump_scare_2"],
            peak_value=tension_peak,
            rise_tau=rng.uniform(90, 180),
            recovery_tau=rng.uniform(90, 210) * recovery_slowdown,
        )

        # Jump scare responses: peak a few seconds after the event and recover slowly.
        js1_amp = rng.uniform(12, 24) * reactivity
        js2_amp = rng.uniform(18, 34) * reactivity

        hr_profile += _alpha_response(
            ecg_time,
            events["jump_scare_1"],
            amplitude=js1_amp,
            delay=rng.uniform(0.8, 1.8),
            rise_tau=rng.uniform(1.5, 3.0),
            decay_tau=rng.uniform(28, 70) * recovery_slowdown,
        )
        # Small slower tail: represents lingering arousal rather than an immediate reset.
        hr_profile += _alpha_response(
            ecg_time,
            events["jump_scare_1"],
            amplitude=js1_amp * rng.uniform(0.15, 0.35),
            delay=rng.uniform(4, 8),
            rise_tau=rng.uniform(8, 18),
            decay_tau=rng.uniform(90, 180) * recovery_slowdown,
        )

        hr_profile += _alpha_response(
            ecg_time,
            events["jump_scare_2"],
            amplitude=js2_amp,
            delay=rng.uniform(0.8, 1.8),
            rise_tau=rng.uniform(1.5, 3.0),
            decay_tau=rng.uniform(35, 90) * recovery_slowdown,
        )
        hr_profile += _alpha_response(
            ecg_time,
            events["jump_scare_2"],
            amplitude=js2_amp * rng.uniform(0.20, 0.45),
            delay=rng.uniform(4, 8),
            rise_tau=rng.uniform(10, 22),
            decay_tau=rng.uniform(120, 240) * recovery_slowdown,
        )

        # Low-frequency physiological HR fluctuation, not per-sample random jumps.
        knot_times = np.arange(0, DURATION + 1, 10)
        knot_noise = rng.normal(0, 1.3, len(knot_times))
        hr_profile += np.interp(ecg_time, knot_times, knot_noise)
        hr_profile = np.clip(hr_profile, 50, 145)

        # Generate ECG waveform from HR profile.
        ecg_signal = np.zeros(ecg_samples)

        # Generate beat times from HR profile.
        beat_times = []
        current_time = 0.5  # start after 0.5s
        while current_time < DURATION:
            idx = min(int(current_time * ECG_SR), ecg_samples - 1)
            current_hr = hr_profile[idx]
            # Natural HRV. Keep bounded so detection remains teachable.
            ibi = 60.0 / current_hr + rng.normal(0, 0.025)
            ibi = max(ibi, 0.42)  # minimum IBI, roughly <=143 bpm
            beat_times.append(current_time)
            current_time += ibi

        # Build ECG waveform around each beat.
        for bt in beat_times:
            bt_idx = int(bt * ECG_SR)
            # P wave (small bump before QRS)
            for di in range(-15, -8):
                idx = bt_idx + di
                if 0 <= idx < ecg_samples:
                    ecg_signal[idx] += 0.15 * np.exp(-0.5 * ((di + 11) / 2) ** 2)
            # QRS complex
            for di in range(-3, 4):
                idx = bt_idx + di
                if 0 <= idx < ecg_samples:
                    if di == 0:
                        ecg_signal[idx] += 1.0 + rng.normal(0, 0.05)  # R peak
                    elif di == -1:
                        ecg_signal[idx] += -0.15  # Q
                    elif di == 1:
                        ecg_signal[idx] += -0.2   # S
                    else:
                        ecg_signal[idx] += 0.02
            # T wave
            for di in range(8, 20):
                idx = bt_idx + di
                if 0 <= idx < ecg_samples:
                    ecg_signal[idx] += 0.25 * np.exp(-0.5 * ((di - 13) / 3) ** 2)

        # Add noise.
        ecg_noise = rng.normal(0, 0.03, ecg_samples)
        baseline_wander = 0.1 * np.sin(2 * np.pi * ecg_time * 0.15)
        baseline_wander += 0.05 * np.sin(2 * np.pi * ecg_time * 0.3)

        # Occasional motion artefacts.
        n_motion = rng.integers(2, 6)
        for _ in range(n_motion):
            art_start = rng.integers(0, ecg_samples - 200)
            art_len = rng.integers(50, 200)
            ecg_noise[art_start:art_start + art_len] += rng.normal(
                0, 0.15 * motion_proneness, art_len
            )

        ecg_signal += ecg_noise + baseline_wander

        ecg_df = pd.DataFrame({
            "timestamp": ecg_time,
            "ecg_raw": ecg_signal,
        })
        ecg_df.to_csv(os.path.join(player_dir, "ecg.csv"), index=False)

        # =============================================
        # Generate Facial Action Units (30 Hz)
        # =============================================
        au_samples = DURATION * AU_SR
        au_time = np.arange(au_samples) / AU_SR

        # Action Units we'll generate:
        # AU1 (inner brow raise) - surprise, fear
        # AU2 (outer brow raise) - surprise
        # AU4 (brow lowerer) - concentration, fear
        # AU5 (upper lid raiser) - surprise, fear
        # AU12 (lip corner puller) - smile
        # AU20 (lip stretcher) - fear
        # AU25 (lips part) - surprise
        au_names = ["AU01", "AU02", "AU04", "AU05", "AU12", "AU20", "AU25"]
        au_data = {}

        for au in au_names:
            au_data[au] = rng.exponential(0.1, au_samples).clip(0, 0.5)

        def add_au_response(au_dict, time_arr, event_time, au_activations, duration=5.0, rise=0.5):
            """Add AU activation around an event."""
            for au_name, amplitude in au_activations.items():
                for i, t in enumerate(time_arr):
                    dt = t - event_time
                    if -0.5 < dt < duration:
                        if dt < rise:
                            val = amplitude * max(0, (dt + 0.5) / (rise + 0.5))
                        else:
                            val = amplitude * np.exp(-(dt - rise) / (duration * 0.4))
                        au_dict[au_name][i] = max(au_dict[au_name][i], val + rng.normal(0, 0.1))

        # Jump scare 1: surprise + fear
        add_au_response(au_data, au_time, events["jump_scare_1"], {
            "AU01": rng.uniform(2.5, 4.0),
            "AU02": rng.uniform(2.0, 3.5),
            "AU05": rng.uniform(2.5, 4.0),
            "AU20": rng.uniform(1.5, 3.0),
            "AU25": rng.uniform(3.0, 4.5),
        }, duration=4.0)

        # Tension period: increased AU4 (concentration/worry) + occasional AU20
        tension_dur = events["jump_scare_2"] - events["tension_start"]
        n_tension_face = rng.integers(5, 10)
        for _ in range(n_tension_face):
            t_face = events["tension_start"] + rng.uniform(0, tension_dur)
            add_au_response(au_data, au_time, t_face, {
                "AU04": rng.uniform(1.0, 2.5),
                "AU01": rng.uniform(0.5, 1.5),
            }, duration=3.0)

        # Jump scare 2: stronger fear response
        add_au_response(au_data, au_time, events["jump_scare_2"], {
            "AU01": rng.uniform(3.0, 4.5),
            "AU02": rng.uniform(2.5, 4.0),
            "AU04": rng.uniform(1.5, 3.0),
            "AU05": rng.uniform(3.0, 5.0),
            "AU20": rng.uniform(2.5, 4.0),
            "AU25": rng.uniform(3.5, 5.0),
        }, duration=5.0)

        # Calm periods: occasional smiles.
        for _ in range(rng.integers(3, 8)):
            t_smile = rng.uniform(0, events["tension_start"] - 30)
            add_au_response(au_data, au_time, t_smile, {
                "AU12": rng.uniform(1.5, 3.0),
            }, duration=2.0)

        # Clip to 0-5 range.
        for au in au_names:
            au_data[au] = np.clip(au_data[au], 0, 5.0)

        # Confidence score (mostly high, some drops).
        confidence = np.ones(au_samples) * rng.uniform(0.85, 0.95)
        n_conf_drops = rng.integers(5, 15)
        for _ in range(n_conf_drops):
            drop_start = rng.integers(0, au_samples - 90)
            drop_len = rng.integers(15, 90)  # 0.5 to 3 seconds
            confidence[drop_start:drop_start + drop_len] = rng.uniform(0.1, 0.5, drop_len)
        n_failures = rng.integers(2, 6)
        for _ in range(n_failures):
            fail_start = rng.integers(0, au_samples - 30)
            fail_len = rng.integers(5, 30)
            confidence[fail_start:fail_start + fail_len] = 0.0
            for au in au_names:
                au_data[au][fail_start:fail_start + fail_len] = 0.0

        face_df = pd.DataFrame({"timestamp": au_time, "confidence": confidence})
        for au in au_names:
            face_df[au] = au_data[au]
        face_df.to_csv(os.path.join(player_dir, "face_au.csv"), index=False)

    # Save event ground truth (for lecturer only).
    events_out = {}
    for p in range(n_players):
        pid = f"player_{p+1:03d}"
        events_out[pid] = {k: round(v, 2) for k, v in player_events[p].items()}

    with open(os.path.join(output_dir, "ground_truth_events.json"), "w") as f:
        json.dump(events_out, f, indent=2)

    print(f"Generated data for {n_players} players in '{output_dir}/'")
    print(f"  EDA:  {DURATION * EDA_SR:,} samples per player ({EDA_SR} Hz)")
    print(f"  ECG:  {DURATION * ECG_SR:,} samples per player ({ECG_SR} Hz)")
    print(f"  Face: {DURATION * AU_SR:,} samples per player ({AU_SR} Hz)")
    print("  Ground truth events saved to ground_truth_events.json")


if __name__ == "__main__":
    generate_dataset()
