"""Round-trip + fence test for the 0014 QSA indexer host mirror (#39893 port).

Adapted from the upstream PR's test_qsa_hicache.py to the fork's API:
- fork engine: submit_host_to_device(transfers, layer_num=..., on_layer_done=...)
- fork LayerLoadingEvent.complete(layer_index)
- GPU-stream-level fence (the accessor waits via wait_event; not a host block)

Runs on one GPU alongside the server (tiny allocations).
"""
import sys

import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.qsa import QSAIndexerPoolHost
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

PAGE = 64
RATIO = 4
DEV = "cuda"
NLAYERS = 2  # local (packed) layers; global ids 0,1 with start_layer=0


def make_device_pool():
    return QSATokenToKVPool(
        size=PAGE * 8,
        dtype=torch.bfloat16,
        page_size=PAGE,
        head_num=1,
        head_dim=128,
        full_attention_layer_ids=[0, 1],
        start_layer=0,
        device=DEV,
        mamba_pool=None,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=128,
        qsa_compress_ratio=RATIO,
        qsa_token_topk=128,
        num_request_slots=4,
    )


def slots(pages, device="cpu"):
    return (
        torch.tensor(pages, device=device)[:, None] * PAGE
        + torch.arange(PAGE, device=device)
    ).reshape(-1)


def compressed_slots(indices):
    return indices[::RATIO] // RATIO


def main():
    device = make_device_pool()
    anchor = MHATokenToKVPoolHost(device.full_kv_pool, 2, 64, PAGE, "layer_first")
    host = QSAIndexerPoolHost(device, anchor)
    print("host pool: layer_num=%d item_bytes=%d size=%d" % (
        host.layer_num, host.item_bytes, host.size))

    source = slots([3, 1], DEV)
    destination = slots([4, 2], DEV)
    # kernel backend wants the index tensors on CUDA
    host_indices = slots([2, 0], DEV)

    torch.manual_seed(42)
    device.qsa_compressed_flat.normal_()
    expected = [
        b[compressed_slots(source)].clone()
        for b in device.qsa_compressed_k_buffer_pool
    ]

    engine = L2TransferEngine("kernel")
    engine.submit_device_to_host(
        [
            L2Transfer(
                host_pool=host,
                device_pool=device,
                host_indices=host_indices,
                device_indices=source,
                layer_mapper=lambda l: l,
            )
        ]
    ).finish_event.synchronize()
    print("D2H backup ok")

    # --- H2D restore behind a long sleep, WITH the consumer fence: the
    # accessor must wait per-layer for the in-flight copy.
    device.qsa_compressed_flat.fill_(-99)
    counter = LayerDoneCounter(NLAYERS)
    producer = counter.update_producer()
    counter.set_consumer(producer)
    device.register_layer_transfer_counter(counter)

    with torch.cuda.stream(engine.host_to_device_stream):
        torch.cuda._sleep(20_000_000)
    engine.submit_host_to_device(
        [
            L2Transfer(
                host_pool=host,
                device_pool=device,
                host_indices=host_indices,
                device_indices=destination,
                layer_mapper=lambda l: l,
            )
        ],
        layer_num=NLAYERS,
        on_layer_done=counter.events[producer].complete,
    )
    # Read through the PRODUCTION accessor (fenced) on the default stream,
    # without a global synchronize.
    actual = [
        device.get_qsa_compressed_k_buffer(layer)[compressed_slots(destination)]
        for layer in (0, 1)
    ]
    for restored, saved in zip(actual, expected):
        torch.testing.assert_close(restored, saved, rtol=0, atol=0)
    print("ROUNDTRIP + FENCE bit-exact ok")

    # --- negative control: same scenario WITHOUT the fence must NOT be
    # bit-exact (proves the test actually detects the race).
    device.qsa_compressed_flat.fill_(-99)
    counter2 = LayerDoneCounter(NLAYERS)
    p2 = counter2.update_producer()
    counter2.set_consumer(p2)
    device.register_layer_transfer_counter(None)  # no fence
    with torch.cuda.stream(engine.host_to_device_stream):
        torch.cuda._sleep(20_000_000)
    engine.submit_host_to_device(
        [
            L2Transfer(
                host_pool=host,
                device_pool=device,
                host_indices=host_indices,
                device_indices=destination,
                layer_mapper=lambda l: l,
            )
        ],
        layer_num=NLAYERS,
    )
    unfenced = [
        device.get_qsa_compressed_k_buffer(layer)[compressed_slots(destination)]
        for layer in (0, 1)
    ]
    import copy as _copy

    # read on the default stream WITHOUT the fence: data may be garbage
    # (the sleep keeps the H2D stream busy). We only require that the
    # fenced read above was meaningful; the unfenced read may or may not
    # race (timing-dependent), so print, don't assert.
    try:
        for restored, saved in zip(unfenced, expected):
            torch.testing.assert_close(restored, saved, rtol=0, atol=0)
        print("unfenced control: no race observed this run (timing-dependent)")
    except AssertionError:
        print("unfenced control: race observed (stale data) - as expected")

    assert host.size == anchor.size, (host.size, anchor.size)
    assert host.size_per_token * PAGE == sum(
        x[0].nbytes for x in host.device_buffers
    ), (host.size_per_token, host.device_buffers)
    print("size-parity ok (host.size == anchor.size == %d)" % host.size)
    print("ALL-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
