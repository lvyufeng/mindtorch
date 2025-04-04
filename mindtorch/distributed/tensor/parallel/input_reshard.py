# Copyright (c) Meta Platforms, Inc. and affiliates
from functools import partial
from typing import Any, Optional, Tuple

import mindtorch
from mindtorch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard


__all__ = [
    "input_reshard",
]


def input_reshard(
    module: mindtorch.nn.Module,
    tp_device_mesh: DeviceMesh,
    input_reshard_dim: Optional[int] = None,
) -> mindtorch.nn.Module:
    """
    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.

    Register hooks to an nn.Module with input resharding so that we can shard
    per the given `tp_device_mesh` and `input_reshard_dim` and restore the
    input back when recomputing the activations in the backward. The reason
    why we can do this is that for Tensor Parallel(TP), the input are same
    across all TP ranks.

    Args:
        module (:class:`nn.Module`):
            Module to be registered with input resharding.
        tp_device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for Tensor Parallel.
        input_reshard_dim (Optional[int]):
            The dimension of where we perform the sharding
            of input. If set None, there is no sharding of input.
            Default: None

    Return:
        A :class:`nn.Module` object registered with TP input resharding.
    """
    if input_reshard_dim is None:
        return module

    cx: Optional[mindtorch.autograd.graph.saved_tensors_hooks] = None

    def input_reshard_forward_pre_hook(_: mindtorch.nn.Module, _i: Tuple[Any, ...]) -> None:
        saved_tensor_hooks = mindtorch.autograd.graph.saved_tensors_hooks(
            partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim),
            partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim),
        )
        saved_tensor_hooks.__enter__()
        nonlocal cx
        cx = saved_tensor_hooks  # type: ignore[name-defined]

    def input_reshard_backward_hook(
        _: mindtorch.nn.Module, _i: Tuple[Any, ...], _o: Any
    ) -> Any:
        nonlocal cx
        cx.__exit__()  # type: ignore[name-defined, union-attr]

    module.register_forward_pre_hook(input_reshard_forward_pre_hook)
    module.register_forward_hook(input_reshard_backward_hook)
    return module


def _pack_hook_tp(
    mesh: DeviceMesh, input_reshard_dim: int, x: mindtorch.Tensor
) -> Any:  # noqa: D401
    """Hook function called after FWD to shard input."""
    if isinstance(x, DTensor) and all(p.is_replicate() for p in x._spec.placements):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, mindtorch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(x, device_mesh=mesh)
            .redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
            .to_local()
        )
    else:
        return x


def _unpack_hook_tp(
    mesh: DeviceMesh, input_reshard_dim: int, x: Any
) -> mindtorch.Tensor:  # noqa: D401
    """Hook function called before activation recomputing in BWD to restore input."""
    if (
        isinstance(x, DTensor)
        and len(x._spec.placements) == 1
        and x._spec.placements[0].is_shard()
    ):
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, mindtorch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(
                x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]
            )
            .redistribute(device_mesh=mesh, placements=[Replicate()])
            .to_local()
        )
    else:
        return x
