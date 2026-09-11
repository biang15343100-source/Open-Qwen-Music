from __future__ import annotations

import math

import pytest
import torch

from open_qwen_music.render.flow import (
    ConsistencyFlowMatchingConfig,
    FlowConfig,
    channelwise_mse_sums,
    classifier_free_guidance,
    couple_minibatch_source,
    euler_solve,
    heun_solve,
    linear_flow_path,
    consistency_flow_matching_loss,
    make_consistency_flow_matching_pair,
    make_flow_training_sample,
    masked_flow_loss,
    sample_source_like,
    sample_timesteps,
    sampling_interval,
    sampling_schedule,
    shift_timesteps_by_length,
    solve_flow,
    timestep_channelwise_mse_sums,
)


def test_consistency_flow_matching_config_rejects_unknown_fields() -> None:
    config = ConsistencyFlowMatchingConfig.from_mapping(
        {
            "enabled": True,
            "delta": 1.0e-3,
            "num_segments": 2,
            "boundary": 0.9,
            "boundary_zero_steps": 250,
            "velocity_weight": 1.0e-5,
        }
    )
    assert config.enabled
    assert config.for_step(249).boundary == 0.0
    assert config.for_step(250).boundary == pytest.approx(0.9)
    with pytest.raises(ValueError, match="contains unknown fields"):
        ConsistencyFlowMatchingConfig.from_mapping({"enabled": True, "alpha": 1})


def test_consistency_pair_maps_data_to_noise_into_author_progress() -> None:
    source = torch.full((4, 1, 1), 10.0)
    target = torch.full((4, 1, 1), 20.0)
    flow = FlowConfig(time_direction="data_to_noise")
    sample = make_flow_training_sample(
        target,
        flow,
        source=source,
        timestep=torch.tensor([0.75, 0.60, 0.50, 0.10]),
        apply_source_coupling=False,
    )
    consistency = ConsistencyFlowMatchingConfig(
        enabled=True,
        delta=0.1,
        num_segments=2,
        boundary=0.9,
    )
    pair = make_consistency_flow_matching_pair(sample, flow, consistency)
    torch.testing.assert_close(
        pair.neighbor_timestep,
        torch.tensor([0.65, 0.50, 0.40, 0.00]),
    )
    torch.testing.assert_close(
        pair.segment_endpoint_timestep,
        torch.tensor([0.50, 0.50, 0.50, 0.00]),
    )
    torch.testing.assert_close(
        pair.segment_endpoint_latents[:, 0, 0],
        torch.tensor([15.0, 15.0, 15.0, 20.0]),
    )
    assert pair.use_predicted_neighbor_endpoint.tolist() == [True, True, True, False]
    assert pair.velocity_consistency_active.tolist() == [True, False, False, False]
    assert pair.requires_neighbor_prediction


def test_consistency_boundary_zero_uses_true_segment_endpoint() -> None:
    source = torch.zeros(1, 2, 1)
    target = torch.ones_like(source)
    flow = FlowConfig(time_direction="data_to_noise")
    sample = make_flow_training_sample(
        target,
        flow,
        source=source,
        timestep=torch.tensor([0.75]),
        apply_source_coupling=False,
    )
    consistency = ConsistencyFlowMatchingConfig(
        enabled=True,
        delta=0.01,
        num_segments=2,
        boundary=0.0,
    )
    pair = make_consistency_flow_matching_pair(sample, flow, consistency)
    assert not pair.requires_neighbor_prediction
    prediction = sample.target_velocity.detach().clone().requires_grad_(True)
    result = consistency_flow_matching_loss(
        prediction,
        None,
        sample,
        pair,
        torch.ones(1, 2, dtype=torch.bool),
        flow_config=flow,
        consistency_config=consistency,
    )
    assert result.endpoint.loss.item() == pytest.approx(0.0)
    assert result.velocity.loss.item() == pytest.approx(0.0)
    result.combined.numerator.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_consistency_neighbor_is_stop_gradient_and_inactive_velocity_keeps_denominator() -> None:
    source = torch.zeros(2, 1, 1)
    target = torch.ones_like(source)
    flow = FlowConfig(time_direction="data_to_noise")
    sample = make_flow_training_sample(
        target,
        flow,
        source=source,
        timestep=torch.tensor([0.75, 0.10]),
        apply_source_coupling=False,
    )
    consistency = ConsistencyFlowMatchingConfig(
        enabled=True,
        delta=0.01,
        num_segments=2,
        boundary=0.9,
        velocity_weight=1.0e-5,
    )
    pair = make_consistency_flow_matching_pair(sample, flow, consistency)
    primary = torch.zeros_like(target, requires_grad=True)
    neighbor = torch.full_like(target, 2.0, requires_grad=True)
    result = consistency_flow_matching_loss(
        primary,
        neighbor,
        sample,
        pair,
        torch.ones(2, 1, dtype=torch.bool),
        flow_config=flow,
        consistency_config=consistency,
    )

    assert result.velocity.denominator.item() == pytest.approx(2.0)
    assert result.velocity.loss.item() == pytest.approx(2.0)
    result.combined.numerator.backward()
    assert primary.grad is not None and torch.count_nonzero(primary.grad) > 0
    assert neighbor.grad is None


def test_linear_rectified_flow_sign_and_endpoints() -> None:
    source = torch.tensor([[[1.0, -2.0]]])
    target = torch.tensor([[[4.0, 2.0]]])
    at_zero, velocity = linear_flow_path(source, target, torch.tensor([0.0]))
    at_one, _ = linear_flow_path(source, target, torch.tensor([1.0]))
    assert torch.equal(at_zero, source)
    assert torch.equal(at_one, target)
    assert torch.equal(velocity, target - source)


def test_fulldit_data_to_noise_path_and_reverse_sampling_interval() -> None:
    noise = torch.tensor([[[1.0, -2.0]]])
    data = torch.tensor([[[4.0, 2.0]]])
    at_zero, velocity = linear_flow_path(
        noise,
        data,
        torch.tensor([0.0]),
        time_direction="data_to_noise",
    )
    at_one, _ = linear_flow_path(
        noise,
        data,
        torch.tensor([1.0]),
        time_direction="data_to_noise",
    )
    assert torch.equal(at_zero, data)
    assert torch.equal(at_one, noise)
    assert torch.equal(velocity, noise - data)
    assert sampling_interval(FlowConfig(time_direction="data_to_noise")) == (1.0, 0.0)


def test_unknown_time_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="time_direction"):
        FlowConfig(time_direction="sideways").validate()


def test_masked_loss_uses_explicit_numerator_denominator() -> None:
    prediction = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    result = masked_flow_loss(prediction, target, mask)
    assert result.numerator.item() == pytest.approx(5.0)
    assert result.denominator.item() == pytest.approx(1.0)
    assert result.loss.item() == pytest.approx(5.0)

    prediction[:, 1] = -1.0e9
    assert masked_flow_loss(prediction, target, mask).loss.item() == pytest.approx(5.0)
    with pytest.raises(ValueError, match="valid frame with positive weight"):
        masked_flow_loss(prediction, target, torch.zeros_like(mask))


def test_flow_compute_and_reduction_semantics_are_explicit() -> None:
    config = FlowConfig(
        loss_reduction="valid_frame_mean",
        source_compute_dtype="float32",
        timestep_compute_dtype="float32",
        loss_compute_dtype="float32",
    )
    config.validate()
    source = sample_source_like(
        torch.empty(2, 3, dtype=torch.float64),
        config,
        generator=torch.Generator().manual_seed(7),
    )
    assert source.dtype == torch.float64
    expected = torch.randn(
        2,
        3,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(7),
    ).double()
    assert torch.equal(source, expected)

    prediction = torch.tensor(
        [
            [[1.0], [1.0], [1.0]],
            [[3.0], [0.0], [0.0]],
        ]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    result = masked_flow_loss(
        prediction,
        target,
        mask,
        loss_reduction="valid_frame_mean",
        compute_dtype="float32",
    )

    assert result.loss.item() == pytest.approx(3.0)

    sample_mean = masked_flow_loss(
        prediction,
        target,
        mask,
        loss_reduction="sample_mean",
        compute_dtype="float32",
    )
    assert sample_mean.numerator.item() == pytest.approx(10.0)
    assert sample_mean.denominator.item() == pytest.approx(2.0)
    assert sample_mean.loss.item() == pytest.approx(5.0)
    FlowConfig(loss_reduction="sample_mean").validate()
    with pytest.raises(ValueError, match="source_compute_dtype"):
        FlowConfig(source_compute_dtype="float64").validate()


def test_sample_mean_reduction_is_microbatch_additive() -> None:
    prediction = torch.tensor(
        [
            [[1.0], [1.0], [1.0]],
            [[3.0], [0.0], [0.0]],
            [[2.0], [2.0], [0.0]],
        ]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor(
        [
            [True, True, True],
            [True, False, False],
            [True, True, False],
        ]
    )
    full = masked_flow_loss(
        prediction,
        target,
        mask,
        loss_reduction="sample_mean",
    )
    left = masked_flow_loss(
        prediction[:2],
        target[:2],
        mask[:2],
        loss_reduction="sample_mean",
    )
    right = masked_flow_loss(
        prediction[2:],
        target[2:],
        mask[2:],
        loss_reduction="sample_mean",
    )
    torch.testing.assert_close(
        full.numerator,
        left.numerator + right.numerator,
    )
    torch.testing.assert_close(
        full.denominator,
        left.denominator + right.denominator,
    )
    assert full.loss.item() == pytest.approx((1.0 + 9.0 + 4.0) / 3.0)


def test_sample_mean_respects_sample_level_weights() -> None:
    prediction = torch.tensor(
        [
            [[1.0], [1.0], [1.0]],
            [[3.0], [0.0], [0.0]],
        ]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    result = masked_flow_loss(
        prediction,
        target,
        mask,
        sample_weight=torch.tensor([1.0, 3.0]),
        loss_reduction="sample_mean",
    )
    assert result.numerator.item() == pytest.approx(28.0)
    assert result.denominator.item() == pytest.approx(4.0)
    assert result.loss.item() == pytest.approx(7.0)


def test_channelwise_mse_treats_every_latent_dimension_uniformly() -> None:
    prediction = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]],
            [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
        ]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False], [True, True]])
    numerator, denominator = channelwise_mse_sums(prediction, target, mask)
    torch.testing.assert_close(
        numerator,
        torch.tensor([14.0, 56.0, 126.0], dtype=torch.float64),
    )
    assert denominator.item() == 3.0


def test_timestep_channelwise_mse_has_independent_bin_denominators() -> None:
    prediction = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 4.0]],
            [[3.0, 6.0], [9.0, 9.0]],
        ]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True], [True, False]])
    timestep = torch.tensor([0.05, 0.95])
    numerator, denominator = timestep_channelwise_mse_sums(
        prediction,
        target,
        mask,
        timestep,
        num_bins=2,
    )
    torch.testing.assert_close(
        numerator,
        torch.tensor([[5.0, 20.0], [9.0, 36.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        denominator,
        torch.tensor([2.0, 1.0], dtype=torch.float64),
    )


@pytest.mark.parametrize("solver", [euler_solve, heun_solve])
def test_constant_vector_field_has_correct_sign_and_is_exact(solver) -> None:
    initial = torch.randn(2, 5, 3)
    constant = torch.tensor([0.5, -1.0, 2.0]).view(1, 1, 3)

    def field(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return constant.expand_as(state)

    result = solver(field, initial, num_steps=7)
    assert torch.allclose(result, initial + constant, atol=1.0e-6)


def test_euler_first_order_and_heun_second_order_convergence() -> None:
    initial = torch.ones(1, 1, 1, dtype=torch.float64)

    def field(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return state

    exact = math.e
    euler_8 = abs(euler_solve(field, initial, num_steps=8).item() - exact)
    euler_16 = abs(euler_solve(field, initial, num_steps=16).item() - exact)
    heun_8 = abs(heun_solve(field, initial, num_steps=8).item() - exact)
    heun_16 = abs(heun_solve(field, initial, num_steps=16).item() - exact)
    assert euler_8 / euler_16 > 1.8
    assert heun_8 / heun_16 > 3.5
    assert heun_8 < euler_8


@pytest.mark.parametrize("solver", [euler_solve, heun_solve])
def test_solver_supports_reverse_time(solver) -> None:
    initial = torch.tensor([[[2.0]]])

    def field(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(state)

    result = solver(field, initial, num_steps=4, t_start=1.0, t_end=0.0)
    torch.testing.assert_close(result, torch.tensor([[[1.0]]]))


def test_heun_fractional_reverse_interval_uses_exact_endpoint() -> None:
    initial = torch.ones(1, 1, 1)
    observed: list[float] = []

    def field(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        assert bool(((timestep >= 0.0) & (timestep <= 1.0)).all())
        observed.extend(float(value) for value in timestep)
        return torch.zeros_like(state)

    result = heun_solve(
        field,
        initial,
        num_steps=16,
        t_start=0.95,
        t_end=0.0,
    )
    torch.testing.assert_close(result, initial)
    assert observed[-1] == 0.0


def test_cfg_boundary_values() -> None:
    conditional = torch.randn(2, 3, 4)
    null = torch.randn(2, 3, 4)
    assert torch.equal(classifier_free_guidance(conditional, null, 0.0), null)
    assert torch.equal(classifier_free_guidance(conditional, null, 1.0), conditional)


def test_timestep_distributions_and_solver_nfe_are_explicit() -> None:
    reference = torch.empty(4)
    config = FlowConfig(timestep_distribution="logit_normal")
    first = sample_timesteps(
        4,
        config,
        reference=reference,
        generator=torch.Generator().manual_seed(7),
    )
    second = sample_timesteps(
        4,
        config,
        reference=reference,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(first, second)
    assert bool(((first > 0) & (first < 1)).all())

    mixture = FlowConfig(
        timestep_distribution="uniform_logit_normal_50_50",
        logit_normal_mean=1.0,
        logit_normal_std=0.5,
    )
    mixed_first = sample_timesteps(
        4_096,
        mixture,
        reference=reference,
        generator=torch.Generator().manual_seed(11),
    )
    mixed_second = sample_timesteps(
        4_096,
        mixture,
        reference=reference,
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.equal(mixed_first, mixed_second)
    assert bool(((mixed_first > 0) & (mixed_first < 1)).all())


    assert float((mixed_first < 0.1).float().mean()) > 0.03
    assert float((mixed_first > 0.9).float().mean()) > 0.03

    quarter_mixture = FlowConfig(
        timestep_distribution="uniform_logit_normal_25_75",
        logit_normal_mean=1.0,
        logit_normal_std=0.5,
    )
    quarter_values = sample_timesteps(
        4_096,
        quarter_mixture,
        reference=reference,
        generator=torch.Generator().manual_seed(11),
    )

    assert float((quarter_values < 0.1).float().mean()) > 0.01
    assert float((quarter_values > 0.9).float().mean()) > 0.01
    assert float(quarter_values.mean()) > float(mixed_first.mean())

    output = solve_flow(
        lambda state, time: torch.zeros_like(state),
        torch.zeros(1, 2, 3),
        solver="heun",
        num_steps=5,
    )
    assert output.nfe == 10
    with pytest.raises(ValueError, match="solver"):
        solve_flow(
            lambda state, time: state,
            torch.zeros(1, 2, 3),
            solver="implicit",
            num_steps=1,
        )


def test_stable_audio_truncated_timestep_and_length_shift_are_explicit() -> None:
    config = FlowConfig(
        time_direction="data_to_noise",
        timestep_distribution="truncated_logit_normal_rescaled",
        timestep_shift="length_logistic",
        timestep_shift_min_frames=100,
        timestep_shift_max_frames=1_000,
        timestep_shift_base=0.5,
        timestep_shift_max=1.15,
    )
    first = sample_timesteps(
        4_096,
        config,
        reference=torch.empty(1),
        generator=torch.Generator().manual_seed(19),
        effective_lengths=torch.full((4_096,), 100),
    )
    second = sample_timesteps(
        4_096,
        config,
        reference=torch.empty(1),
        generator=torch.Generator().manual_seed(19),
        effective_lengths=torch.full((4_096,), 100),
    )
    assert torch.equal(first, second)
    assert bool(((first > 0.0) & (first < 1.0)).all())

    base = torch.tensor([0.0, 0.5, 1.0])
    short = shift_timesteps_by_length(base, [100, 100, 100], config)
    long = shift_timesteps_by_length(base, [1_000, 1_000, 1_000], config)
    assert short[0] == 0.0 and short[-1] == 1.0
    assert long[0] == 0.0 and long[-1] == 1.0
    assert float(long[1]) > float(short[1]) > 0.5


def test_length_shifted_sampling_schedule_drives_per_sample_ode_grid() -> None:
    config = FlowConfig(
        time_direction="data_to_noise",
        timestep_shift="length_logistic",
        timestep_shift_min_frames=10,
        timestep_shift_max_frames=100,
    )
    initial = torch.ones(2, 3, 1)
    schedule = sampling_schedule(
        config,
        num_steps=8,
        reference=initial,
        effective_lengths=torch.tensor([10, 100]),
    )
    assert schedule.shape == (2, 9)
    assert torch.equal(schedule[:, 0], torch.ones(2))
    assert torch.equal(schedule[:, -1], torch.zeros(2))
    assert float(schedule[1, 4]) > float(schedule[0, 4])

    result = solve_flow(
        lambda state, timestep: torch.ones_like(state),
        initial,
        solver="heun",
        num_steps=8,
        t_start=1.0,
        t_end=0.0,
        schedule=schedule,
    )
    torch.testing.assert_close(result.sample, torch.zeros_like(initial))


def test_minibatch_ot_reorders_source_and_reduces_masked_cost() -> None:
    target = torch.tensor([[[0.0], [0.0]], [[10.0], [10.0]]])
    source = torch.tensor([[[10.0], [10.0]], [[0.0], [0.0]]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    result = couple_minibatch_source(
        target,
        source,
        mask,
        sinkhorn_iterations=20,
    )
    assert torch.equal(result.permutation, torch.tensor([1, 0]))
    assert torch.equal(result.source, target)
    assert float(result.cost_after) == 0.0
    assert float(result.cost_before) > float(result.cost_after)


def test_minibatch_ot_rejects_silent_batch_one_noop() -> None:
    with pytest.raises(RuntimeError, match="batch<2"):
        couple_minibatch_source(
            torch.zeros(1, 2, 1),
            torch.ones(1, 2, 1),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_global_minibatch_ot_rejects_explicit_local_source() -> None:
    config = FlowConfig(
        source_coupling="minibatch_ot",
        source_coupling_scope="global",
    )
    target = torch.zeros(2, 2, 1)
    with pytest.raises(ValueError, match="does not accept an explicit local source"):
        make_flow_training_sample(
            target,
            config,
            source=torch.ones_like(target),
        )
