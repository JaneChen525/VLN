#!/usr/bin/env python3
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys


def assert_portable_elf(module_name: str) -> None:
    from elftools.elf.elffile import ELFFile

    module = importlib.import_module(module_name)
    path = module.__file__
    dynamic_paths = []
    with open(path, 'rb') as handle:
        elf = ELFFile(handle)
        for segment in elf.iter_segments():
            if segment.header.p_type != 'PT_DYNAMIC':
                continue
            for tag in segment.iter_tags():
                if tag.entry.d_tag in {'DT_RPATH', 'DT_RUNPATH'}:
                    value = getattr(tag, 'rpath', getattr(tag, 'runpath', ''))
                    dynamic_paths.extend(filter(None, value.split(':')))
    absolute = [value for value in dynamic_paths if os.path.isabs(value)]
    print(f'{module_name}_rpath={dynamic_paths}')
    if absolute:
        raise RuntimeError(
            f'{module_name} contains non-portable absolute RPATH entries: {absolute}'
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--expect-model-type', choices=('qwen3_5', 'qwen3_vl'))
    parser.add_argument('--run-kernel', action='store_true')
    args = parser.parse_args()

    import torch
    import transformers
    import vllm

    print(f'python={sys.version.split()[0]} executable={sys.executable}')
    print(f'platform={platform.platform()}')
    print(f'torch={torch.__version__} torch_cuda={torch.version.cuda}')
    print(f'transformers={transformers.__version__} vllm={vllm.__version__}')
    from transformers.modeling_flash_attention_utils import _is_packed_sequence
    regression_position_ids = torch.arange(32).view(1, 1, 32).expand(3, 1, 32)
    if _is_packed_sequence(regression_position_ids, batch_size=1):
        raise RuntimeError(
            'Transformers Qwen3.5/FA2 regression is unpatched: '
            '3-D M-RoPE IDs were classified as a packed sequence'
        )
    print('qwen35_fa2_3d_position_ids_guard=OK')
    print(f'cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the Qwen3.5 training Conda environment')
    capability = torch.cuda.get_device_capability(0)
    required_arch = f'sm_{capability[0]}{capability[1]}'
    torch_arches = torch.cuda.get_arch_list()
    print(
        f'gpu0={torch.cuda.get_device_name(0)} capability={capability} '
        f'torch_arches={torch_arches}'
    )
    if required_arch not in torch_arches:
        raise RuntimeError(
            f'PyTorch wheel lacks device code for {required_arch}: {torch_arches}'
        )

    from causal_conv1d import causal_conv1d_fn
    from flash_attn import flash_attn_func
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    assert_portable_elf('causal_conv1d_cuda')
    assert_portable_elf('flash_attn_2_cuda')
    expected_versions = {
        'causal-conv1d': '1.6.2.post1',
        'flash-attn': '2.8.3',
        'fla-core': '0.4.2',
        'flash-linear-attention': '0.4.2',
    }
    for distribution, expected in expected_versions.items():
        actual = importlib.metadata.version(distribution)
        print(f'{distribution}={actual}')
        if actual != expected:
            raise RuntimeError(
                f'{distribution} version mismatch: expected={expected} actual={actual}'
            )
    print(f'causal_conv1d={importlib.import_module("causal_conv1d").__file__}')
    print(f'flash_attn={importlib.import_module("flash_attn").__version__}')
    print(f'fla={importlib.import_module("fla").__file__}')

    if args.model:
        config_path = os.path.join(args.model, 'config.json')
        with open(config_path, encoding='utf-8') as handle:
            config = json.load(handle)
        text_config = config.get('text_config', config)
        layer_types = text_config.get('layer_types', [])
        print(
            'model_type={} architecture={} linear_layers={} full_layers={}'.format(
                config.get('model_type'),
                config.get('architectures'),
                sum('linear' in layer for layer in layer_types),
                sum('full' in layer for layer in layer_types),
            )
        )
        model_type = config.get('model_type')
        if model_type not in {'qwen3_5', 'qwen3_vl'}:
            raise RuntimeError(f'Expected qwen3_5 or qwen3_vl, got {model_type}')
        if args.expect_model_type and model_type != args.expect_model_type:
            raise RuntimeError(f'Expected {args.expect_model_type}, got {model_type}')

        from transformers import AutoConfig, AutoProcessor
        hf_config = AutoConfig.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=True
        )
        print(
            f'hf_local_load=OK config={type(hf_config).__name__} '
            f'processor={type(processor).__name__}'
        )

    if args.run_kernel:
        dtype = torch.bfloat16
        x = torch.randn(1, 16, 8, device='cuda', dtype=dtype, requires_grad=True)
        weight = torch.randn(16, 4, device='cuda', dtype=dtype, requires_grad=True)
        y = causal_conv1d_fn(x, weight, None, activation='silu')

        aq = torch.randn(1, 8, 2, 64, device='cuda', dtype=dtype, requires_grad=True)
        ak = torch.randn_like(aq, requires_grad=True)
        av = torch.randn_like(aq, requires_grad=True)
        attention_out = flash_attn_func(aq, ak, av, causal=True)

        batch, seq, heads, dim = 1, 8, 2, 64
        q = torch.randn(
            batch, seq, heads, dim, device='cuda', dtype=dtype, requires_grad=True
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        g = torch.rand(
            batch, seq, heads, device='cuda', dtype=torch.float32, requires_grad=True
        ).log()
        beta = torch.rand(
            batch, seq, heads, device='cuda', dtype=dtype, requires_grad=True
        )
        out = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
        if isinstance(out, tuple):
            out = out[0]
        loss = (
            y.float().square().mean()
            + attention_out.float().square().mean()
            + out.float().square().mean()
        )
        loss.backward()
        torch.cuda.synchronize()
        print(
            f'kernel_forward_backward_smoke=OK causal_shape={tuple(y.shape)} '
            f'fa2_shape={tuple(attention_out.shape)} gdn_shape={tuple(out.shape)}'
        )

    print('STACK_CHECK_OK')


if __name__ == '__main__':
    main()
