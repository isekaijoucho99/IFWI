"""Offline CPU tests for GPU utilization advice and conservative preflight."""
import pytest

from experiments.diffusion_prior.author_unet import check_memory_budget, paper_configuration

GiB = 1024**3


def test_32_channel_author_preset_allows_four_micro_samples_when_free_is_15_gib():
    cfg = paper_configuration(32)
    report = check_memory_budget(cfg, 15*GiB, mode='base_train', microbatch=4,
                                 patch_size=128)
    assert report['parameters'] > 50_000_000
    assert report['estimated_bytes'] <= 15*GiB*.80
    # Still reject an unbounded attempt.
    with pytest.raises(MemoryError):
        check_memory_budget(cfg, 15*GiB, mode='base_train', microbatch=8, patch_size=128)


def test_paper_author_full_model_still_refused_on_16g():
    cfg = paper_configuration(128)
    with pytest.raises(MemoryError):
        check_memory_budget(cfg, 16*GiB, mode='base_train', microbatch=1, patch_size=128)
    with pytest.raises(MemoryError):
        check_memory_budget(cfg, 16*GiB, mode='adapt', microbatch=1, patch_size=128)


def test_projection_uses_measurements_and_stops_when_memory_margin_is_too_small():
    from experiments.diffusion_prior.batch_probe import next_batch_is_safe
    assert next_batch_is_safe([(1, 4.3*GiB)], 2, 15*GiB, safety_fraction=.75)
    assert next_batch_is_safe([(1, 4.3*GiB), (2, 5.5*GiB)], 4, 15*GiB,
                              safety_fraction=.75)
    assert not next_batch_is_safe([(1, 8*GiB)], 2, 15*GiB, safety_fraction=.75)
    with pytest.raises(ValueError):
        next_batch_is_safe([(1, 4.3*GiB)], 1, 15*GiB)


def test_selects_best_safe_throughput_not_largest_microbatch():
    from experiments.diffusion_prior.batch_probe import select_fastest
    records = [
        dict(microbatch=1, peak_reserved_bytes=3*GiB, samples_per_second=9.),
        dict(microbatch=2, peak_reserved_bytes=5*GiB, samples_per_second=17.),
        dict(microbatch=4, peak_reserved_bytes=9*GiB, samples_per_second=16.),
    ]
    result = select_fastest(records, free_bytes=15*GiB, effective_batch=16,
                            safety_fraction=.75)
    assert result['microbatch'] == 2
    assert result['accumulate_steps'] == 8
    assert result['effective_batch'] == 16
    smaller = select_fastest(records, free_bytes=8*GiB, effective_batch=16,
                             safety_fraction=.75)
    assert smaller["microbatch"] == 2
    with pytest.raises(ValueError):
        select_fastest(records, free_bytes=3*GiB, effective_batch=16,
                       safety_fraction=.75)


def test_probe_does_not_silently_run_on_cpu():
    from experiments.diffusion_prior.batch_probe import main
    with pytest.raises(SystemExit):
        main(['--device','cpu','--out','unused.json'])
