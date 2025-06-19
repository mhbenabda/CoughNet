#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Test unload() / flatten() software operator
"""
import os
import sys

import numpy as np

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import izer.tornadocnn as tc  # noqa: E402 pylint: disable=wrong-import-position
from izer.utils import ffs, popcount  # noqa: E402 pylint: disable=wrong-import-position

MEM_INVALID = -(2**63)  # When encountering this value, we know the array value was not initialized


def unload(apb_base, processor_map, input_shape,
           out_array, out_offset, in_array, flatten=False):
    """
    Unload HWC memory and return it in `out_array`.
    The generated C code is specific to the network configuration passed in in `processor_map`,
    `input_shape`, and `chan`. Additionally, the generated addresses are offset by `apb_base` and
    `out_offset`. The C code function takes a pointer to a memory array, and the dimensions of
    the array do not matter (flattened or not flattened).
    The additional simulation code takes the `flatten` parameter and an `in_array`.
    If `flatten` is `True`, then the out_array is flattened.
    """
    def get_val(offs):
        """
        Returns value stored at offset `offs` in the memory array.
        """
        if offs >= (tc.dev.MEM_SIZE << 2) or offs < 0:
            raise RuntimeError(f'Offset {offs:04x} is invalid for the memory array.')
        if offs & 3:
            raise RuntimeError(f'Offset {offs:04x} should be a 32-bit address.')
        if in_array[offs >> 2] == MEM_INVALID:
            raise RuntimeError(f'Trying to read from uninitialized memory at location {offs:04x}.')
        return in_array[offs >> 2]

    print('\n// Custom unload for this network:\n'
          f'// Input shape: {input_shape}\n'
          'void unload(uint8_t *out_buf)\n'
          '{\n  uint32_t val, *addr, offs;\n')

    coffs = ffs(processor_map) & ~(tc.dev.P_SHARED-1)
    next_layer_map = processor_map >> coffs
    read_addr = None
    write_addr = None
    c = 0
    while c < input_shape[0]:
        for doffs in range(input_shape[1] * input_shape[2]):
            row, col = divmod(doffs, input_shape[2])
            this_map = next_layer_map
            this_c = c

            # Get four bytes from memory array
            proc = (coffs % tc.dev.MAX_PROC) & ~(tc.dev.P_SHARED-1)
            # FIXME: seq = ...
            offs = out_offset + \
                (((proc % tc.dev.P_NUMPRO) * tc.dev.INSTANCE_SIZE |
                  (proc // tc.dev.P_NUMPRO) * tc.dev.C_GROUP_OFFS // 4) +
                 doffs) * 4

            val = get_val(offs)

            if offs != read_addr:
                print(f'  addr = (uint32_t *) 0x{apb_base + tc.dev.C_SRAM_BASE + offs:08x};')
            print('  val = *addr++;')
            read_addr = offs + 4

            # Singulate bytes, ignoring unused processors
            for shift in range(4):
                addr = this_c * input_shape[1] * input_shape[2] + row * input_shape[1] + col
                if shift == 0:
                    if addr != write_addr:
                        print(f'  offs = 0x{addr:04x};')
                    else:
                        print('  offs++;')
                    write_addr = addr + 1
                if this_map & 1:
                    if not flatten:
                        out_array[this_c][row][col] = val & 0xff
                    else:
                        out_array[addr] = val & 0xff
                    print('  out_buf[offs', end='')
                    if shift > 0:
                        print(f'+0x{0x10 * shift:02x}', end='')
                    print('] = ', end='')
                    if shift == 0:
                        print('val', end='')
                    else:
                        print(f'(val >> {shift * 8})', end='')
                    print(' & 0xff;')
                    this_c += 1
                this_map >>= 1
                val >>= 8

        coffs += 4
        c += popcount(next_layer_map & 0x0f)
        next_layer_map >>= 4

    print('}')


def test_unload():
    """
    test case for unload()
    """
    tc.dev = tc.get_device(85)
    np.set_printoptions(threshold=sys.maxsize, linewidth=80,
                        formatter={'int': lambda x: f'{x:02x}'})

    # Create memory image
    mem_image = np.full(tc.dev.MEM_SIZE, MEM_INVALID, dtype=np.int64)

    # Fill image with known values
    instance = 4 << tc.dev.INSTANCE_SHIFT
    mem_image[0x0000 >> 2] = 0x00540055
    mem_image[0x0004 >> 2] = 0x007f0070
    mem_image[0x0008 >> 2] = 0x0e530345
    mem_image[0x000c >> 2] = 0x0946084e
    mem_image[0x0010 >> 2] = 0x044d0045
    mem_image[0x0014 >> 2] = 0x00630051
    mem_image[0x0018 >> 2] = 0x005e0c33
    mem_image[0x001c >> 2] = 0x043d2f41
    mem_image[0x0020 >> 2] = 0x0900002d
    mem_image[0x0024 >> 2] = 0x00000018
    mem_image[0x0028 >> 2] = 0x001c0000
    mem_image[0x002c >> 2] = 0x00180814
    mem_image[0x0030 >> 2] = 0x00000022
    mem_image[0x0034 >> 2] = 0x00000200
    mem_image[0x0038 >> 2] = 0x0005000f
    mem_image[0x003c >> 2] = 0x0002001e
    mem_image[(instance + 0x0000) >> 2] = 0x2d051a0d
    mem_image[(instance + 0x0004) >> 2] = 0x394e141a
    mem_image[(instance + 0x0008) >> 2] = 0x2039141b
    mem_image[(instance + 0x000c) >> 2] = 0x0c000029
    mem_image[(instance + 0x0010) >> 2] = 0x18130913
    mem_image[(instance + 0x0014) >> 2] = 0x0a6c0000
    mem_image[(instance + 0x0018) >> 2] = 0x004f0000
    mem_image[(instance + 0x001c) >> 2] = 0x001a0000
    mem_image[(instance + 0x0020) >> 2] = 0x00000008
    mem_image[(instance + 0x0024) >> 2] = 0x00500000
    mem_image[(instance + 0x0028) >> 2] = 0x005a0000
    mem_image[(instance + 0x002c) >> 2] = 0x004a0000
    mem_image[(instance + 0x0030) >> 2] = 0x0f190b0e
    mem_image[(instance + 0x0034) >> 2] = 0x225b0c17
    mem_image[(instance + 0x0038) >> 2] = 0x006b030f
    mem_image[(instance + 0x003c) >> 2] = 0x00570903
    mem_image[(2 * instance + 0x0000) >> 2] = 0x381e3b00
    mem_image[(2 * instance + 0x0004) >> 2] = 0x6c233a00
    mem_image[(2 * instance + 0x0008) >> 2] = 0x6c002500
    mem_image[(2 * instance + 0x000c) >> 2] = 0x2d000000
    mem_image[(2 * instance + 0x0010) >> 2] = 0x38432800
    mem_image[(2 * instance + 0x0014) >> 2] = 0x646a1700
    mem_image[(2 * instance + 0x0018) >> 2] = 0x53680500
    mem_image[(2 * instance + 0x001c) >> 2] = 0x2734063d
    mem_image[(2 * instance + 0x0020) >> 2] = 0x10573427
    mem_image[(2 * instance + 0x0024) >> 2] = 0x177f2a50
    mem_image[(2 * instance + 0x0028) >> 2] = 0x0a5a004b
    mem_image[(2 * instance + 0x002c) >> 2] = 0x0028003c
    mem_image[(2 * instance + 0x0030) >> 2] = 0x082d0e07
    mem_image[(2 * instance + 0x0034) >> 2] = 0x00400009
    mem_image[(2 * instance + 0x0038) >> 2] = 0x0a1a0419
    mem_image[(2 * instance + 0x003c) >> 2] = 0x00170809

    expected = np.array([
        [[0x55, 0x70, 0x45, 0x4e],
         [0x45, 0x51, 0x33, 0x41],
         [0x2d, 0x18, 0x00, 0x14],
         [0x22, 0x00, 0x0f, 0x1e]],
        [[0x00, 0x00, 0x03, 0x08],
         [0x00, 0x00, 0x0c, 0x2f],
         [0x00, 0x00, 0x00, 0x08],
         [0x00, 0x02, 0x00, 0x00]],
        [[0x54, 0x7f, 0x53, 0x46],
         [0x4d, 0x63, 0x5e, 0x3d],
         [0x00, 0x00, 0x1c, 0x18],
         [0x00, 0x00, 0x05, 0x02]],
        [[0x00, 0x00, 0x0e, 0x09],
         [0x04, 0x00, 0x00, 0x04],
         [0x09, 0x00, 0x00, 0x00],
         [0x00, 0x00, 0x00, 0x00]],
        [[0x0d, 0x1a, 0x1b, 0x29],
         [0x13, 0x00, 0x00, 0x00],
         [0x08, 0x00, 0x00, 0x00],
         [0x0e, 0x17, 0x0f, 0x03]],
        [[0x1a, 0x14, 0x14, 0x00],
         [0x09, 0x00, 0x00, 0x00],
         [0x00, 0x00, 0x00, 0x00],
         [0x0b, 0x0c, 0x03, 0x09]],
        [[0x05, 0x4e, 0x39, 0x00],
         [0x13, 0x6c, 0x4f, 0x1a],
         [0x00, 0x50, 0x5a, 0x4a],
         [0x19, 0x5b, 0x6b, 0x57]],
        [[0x2d, 0x39, 0x20, 0x0c],
         [0x18, 0x0a, 0x00, 0x00],
         [0x00, 0x00, 0x00, 0x00],
         [0x0f, 0x22, 0x00, 0x00]],
        [[0x00, 0x00, 0x00, 0x00],
         [0x00, 0x00, 0x00, 0x3d],
         [0x27, 0x50, 0x4b, 0x3c],
         [0x07, 0x09, 0x19, 0x09]],
        [[0x3b, 0x3a, 0x25, 0x00],
         [0x28, 0x17, 0x05, 0x06],
         [0x34, 0x2a, 0x00, 0x00],
         [0x0e, 0x00, 0x04, 0x08]],
        [[0x1e, 0x23, 0x00, 0x00],
         [0x43, 0x6a, 0x68, 0x34],
         [0x57, 0x7f, 0x5a, 0x28],
         [0x2d, 0x40, 0x1a, 0x17]],
        [[0x38, 0x6c, 0x6c, 0x2d],
         [0x38, 0x64, 0x53, 0x27],
         [0x10, 0x17, 0x0a, 0x00],
         [0x08, 0x00, 0x0a, 0x00]]
    ], dtype=np.int64)

    flattened = expected.flatten()
    computed = np.empty_like(flattened)
    processor_map = 0x0000000000000fff
    input_shape = (12, 4, 4)  # CHW - match generator
    out_offset = 0
    unload(tc.dev.APB_BASE, processor_map, input_shape,
           computed, out_offset, mem_image, flatten=True)

    print('\n// unload(flatten=True):')
    print("// SUCCESS" if np.array_equal(flattened, computed) else "// *** FAILURE ***")

    assert np.array_equal(expected.flatten(), computed)

    computed = np.empty_like(expected)
    unload(tc.dev.APB_BASE, processor_map, input_shape,
           computed, out_offset, mem_image)

    print('\n// unload():')
    print("// SUCCESS" if np.array_equal(expected, computed) else "// *** FAILURE ***")

    assert np.array_equal(expected, computed)


if __name__ == '__main__':
    test_unload()
