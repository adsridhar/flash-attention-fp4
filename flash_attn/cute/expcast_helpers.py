"""Direct E4M3 encoding and exact decoding for HAO's ExpCast FP8 PV kernel.

Every option below is a complete literal PTX block, with a comment beside
every line. No assembly is generated, spliced, or rewritten at module import.
The CuTe wrappers select one block at compile time; there is no GPU-side branch
between options. Existing helper arguments and defaults are unchanged.

Mapping: scalar fma.rn.f32 or paired fma.rn.f32x2 computes 8 * x + bias.
Encoding: unsigned conversion plus shifts/ORs, or signed conversion plus
saturating u8 packing (which ptxas can fuse into F2IP).
Decoding: construct FP32 bits from integer codes, or use exact native
E4M3 -> FP16 -> FP32 conversion. Codes are byte representations, not the
numeric values of the FP8 weights. Positive finite codes are 0..126.

Operand bindings:
  Encode only: $0 = packed output; $1..$4 = scores; $5 = bias; $6 = cap.
  Encode/decode: $0 = packed output; $1..$4 = FP32 weights;
                 $5..$8 = scores; $9 = bias; $10 = cap.
  Decode only: $0..$3 = FP32 weights; $4 = packed input.

The latest native-decoding path with paired mapping and clamps removed selects
_native_paired_unclamped_fused. The previous integer decoder selects
_packed_unclamped_pack_conversion_fused_assembly. The older shift/OR packing
path is also retained. native_decode takes precedence over packed_conversion.

The kernel validates the upper code bound before selecting remove_clamps=True.
Unsigned conversion or saturating u8 packing handles the lower bound, including
masked -inf. Other configurations retain explicit lower and upper clamps.
"""
import cutlass.cute as cute
from cutlass import Float32, Int32, Constexpr, const_expr
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm


# Encode only: scalar mapping, unsigned conversion, shift/OR packing, clamped.
assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3;                                                  // Temporary integer codes, packed bytes, or FP32 bit patterns.
    fma.rn.f32 f0, $1, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f1, $2, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f2, $3, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f3, $4, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode only: scalar mapping, unsigned conversion, shift/OR packing, unclamped.
_unclamped_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3;                                                  // Temporary integer codes, packed bytes, or FP32 bit patterns.
    fma.rn.f32 f0, $1, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f1, $2, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f2, $3, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f3, $4, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode only: paired mapping, unsigned conversion, shift/OR packing, clamped.
_packed_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3;                                                  // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$5, $5};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$1, $2};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$3, $4};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode only: paired mapping, unsigned conversion, shift/OR packing, unclamped.
_packed_unclamped_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3;                                                  // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$5, $5};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$1, $2};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$3, $4};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode only: scalar mapping, saturating byte packing, clamped.
_pack_conversion_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $1, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f1, $2, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f2, $3, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f3, $4, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode only: scalar mapping, saturating byte packing, unclamped.
_unclamped_pack_conversion_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $1, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f1, $2, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f2, $3, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f3, $4, 0f41000000, $5;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode only: paired mapping, saturating byte packing, clamped.
_paired_pack_conversion_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$5, $5};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$1, $2};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$3, $4};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $6;                                                     // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode only: paired mapping, saturating byte packing, unclamped.
_paired_unclamped_pack_conversion_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$5, $5};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$1, $2};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$3, $4};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: scalar mapping, direct integer-code decoding, clamped.
_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,bits;                                             // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $5, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f1, $6, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f2, $7, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f3, $8, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode and decode: scalar mapping, direct integer-code decoding, unclamped.
_unclamped_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,bits;                                             // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $5, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f1, $6, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f2, $7, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    fma.rn.f32 f3, $8, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode and decode: paired mapping, direct integer-code decoding, clamped.
_packed_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,bits;                                             // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$9, $9};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$5, $6};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$7, $8};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode and decode: paired mapping, direct integer-code decoding, unclamped.
_packed_unclamped_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,bits;                                             // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$9, $9};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$5, $6};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$7, $8};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    cvt.rni.u32.f32 c0, f0;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    cvt.rni.u32.f32 c1, f1;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    cvt.rni.u32.f32 c2, f2;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    cvt.rni.u32.f32 c3, f3;                                                 // Round to an unsigned integer code; ties choose even, negatives become zero.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    shl.b32 c1,c1,8;                                                        // Move this code into byte position 1 of the packed word.
    shl.b32 c2,c2,16;                                                       // Move this code into byte position 2 of the packed word.
    shl.b32 c3,c3,24;                                                       // Move this code into byte position 3 of the packed word.
    or.b32 c0,c0,c1;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 c0,c0,c2;                                                        // Combine the next code byte with the bytes already packed.
    or.b32 $0,c0,c3;                                                        // Combine the final byte and return all four packed FP8 codes.
}                                                                           // End this private register scope.
"""

# Encode and decode: scalar mapping, saturating packing, byte/bit decoding, clamped.
_pack_conversion_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $5, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f1, $6, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f2, $7, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f3, $8, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    bfe.u32 c0, packed, 0, 8;                                               // Extract byte 0 as an unsigned integer code.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c1, packed, 8, 8;                                               // Extract byte 1 as an unsigned integer code.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c2, packed, 16, 8;                                              // Extract byte 2 as an unsigned integer code.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c3, packed, 24, 8;                                              // Extract byte 3 as an unsigned integer code.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: scalar mapping, saturating packing, byte/bit decoding, unclamped.
_unclamped_pack_conversion_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $5, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f1, $6, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f2, $7, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f3, $8, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    bfe.u32 c0, packed, 0, 8;                                               // Extract byte 0 as an unsigned integer code.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c1, packed, 8, 8;                                               // Extract byte 1 as an unsigned integer code.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c2, packed, 16, 8;                                              // Extract byte 2 as an unsigned integer code.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c3, packed, 24, 8;                                              // Extract byte 3 as an unsigned integer code.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: paired mapping, saturating packing, byte/bit decoding, clamped.
_packed_pack_conversion_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$9, $9};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$5, $6};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$7, $8};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    bfe.u32 c0, packed, 0, 8;                                               // Extract byte 0 as an unsigned integer code.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c1, packed, 8, 8;                                               // Extract byte 1 as an unsigned integer code.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c2, packed, 16, 8;                                              // Extract byte 2 as an unsigned integer code.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c3, packed, 24, 8;                                              // Extract byte 3 as an unsigned integer code.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: paired mapping, saturating packing, byte/bit decoding, unclamped.
_packed_unclamped_pack_conversion_fused_assembly = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$9, $9};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$5, $6};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$7, $8};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    bfe.u32 c0, packed, 0, 8;                                               // Extract byte 0 as an unsigned integer code.
    add.u32 bits, c0, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c0, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c1, packed, 8, 8;                                               // Extract byte 1 as an unsigned integer code.
    add.u32 bits, c1, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c1, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c2, packed, 16, 8;                                              // Extract byte 2 as an unsigned integer code.
    add.u32 bits, c2, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c2, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c3, packed, 24, 8;                                              // Extract byte 3 as an unsigned integer code.
    add.u32 bits, c3, 960;                                                  // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $4, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c3, 8;                                                 // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $4, $4, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: scalar mapping, saturating packing, native decoding, clamped.
_native_fused = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $5, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f1, $6, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f2, $7, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f3, $8, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    {                                                                       // Begin a private register scope.
        .reg .b32 native_source,half01,half23;                              // Saved packed bytes and the two converted FP16 pairs.
        .reg .b16 byte01,byte23,h0,h1,h2,h3;                                // Containers for FP8 byte pairs and individual FP16 values.
        mov.b32 native_source, packed;                                      // Save the packed input bytes before writing decoded outputs.
        mov.b32 {byte01,byte23}, native_source;                             // Split the packed word into lower and upper pairs of FP8 bytes.
        cvt.rn.f16x2.e4m3x2 half01, byte01;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        cvt.rn.f16x2.e4m3x2 half23, byte23;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        mov.b32 {h0,h1}, half01;                                            // Expose the lower and upper FP16 values separately.
        mov.b32 {h2,h3}, half23;                                            // Expose the lower and upper FP16 values separately.
        cvt.f32.f16 $1, h0;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $2, h1;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $3, h2;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $4, h3;                                                 // Widen one FP16 value exactly to the FP32 weight output.
    }                                                                       // End this private register scope.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: scalar mapping, saturating packing, native decoding, unclamped.
_native_unclamped_fused = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    fma.rn.f32 f0, $5, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f1, $6, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f2, $7, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    fma.rn.f32 f3, $8, 0f41000000, $9;                                      // Compute 8 * score + bias for this score.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    {                                                                       // Begin a private register scope.
        .reg .b32 native_source,half01,half23;                              // Saved packed bytes and the two converted FP16 pairs.
        .reg .b16 byte01,byte23,h0,h1,h2,h3;                                // Containers for FP8 byte pairs and individual FP16 values.
        mov.b32 native_source, packed;                                      // Save the packed input bytes before writing decoded outputs.
        mov.b32 {byte01,byte23}, native_source;                             // Split the packed word into lower and upper pairs of FP8 bytes.
        cvt.rn.f16x2.e4m3x2 half01, byte01;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        cvt.rn.f16x2.e4m3x2 half23, byte23;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        mov.b32 {h0,h1}, half01;                                            // Expose the lower and upper FP16 values separately.
        mov.b32 {h2,h3}, half23;                                            // Expose the lower and upper FP16 values separately.
        cvt.f32.f16 $1, h0;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $2, h1;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $3, h2;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $4, h3;                                                 // Widen one FP16 value exactly to the FP32 weight output.
    }                                                                       // End this private register scope.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: paired mapping, saturating packing, native decoding, clamped.
_native_paired_fused = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$9, $9};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$5, $6};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$7, $8};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    max.f32 f0, f0, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f0, f0, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f1, f1, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f1, f1, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f2, f2, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f2, f2, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    max.f32 f3, f3, 0f00000000;                                             // Clamp negative mapped codes to zero.
    min.f32 f3, f3, $10;                                                    // Clamp mapped codes to the supplied upper code limit.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    {                                                                       // Begin a private register scope.
        .reg .b32 native_source,half01,half23;                              // Saved packed bytes and the two converted FP16 pairs.
        .reg .b16 byte01,byte23,h0,h1,h2,h3;                                // Containers for FP8 byte pairs and individual FP16 values.
        mov.b32 native_source, packed;                                      // Save the packed input bytes before writing decoded outputs.
        mov.b32 {byte01,byte23}, native_source;                             // Split the packed word into lower and upper pairs of FP8 bytes.
        cvt.rn.f16x2.e4m3x2 half01, byte01;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        cvt.rn.f16x2.e4m3x2 half23, byte23;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        mov.b32 {h0,h1}, half01;                                            // Expose the lower and upper FP16 values separately.
        mov.b32 {h2,h3}, half23;                                            // Expose the lower and upper FP16 values separately.
        cvt.f32.f16 $1, h0;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $2, h1;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $3, h2;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $4, h3;                                                 // Widen one FP16 value exactly to the FP32 weight output.
    }                                                                       // End this private register scope.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Encode and decode: paired mapping, saturating packing, native decoding, unclamped.
# Latest optimized path: paired mapping, F2IP-compatible packing, native decoding.
_native_paired_unclamped_fused = """
{                                                                           // Begin a private register scope.
    .reg .f32 f0,f1,f2,f3;                                                  // Temporary FP32 values after mapping scores to byte codes.
    .reg .b32 c0,c1,c2,c3,packed,bits;                                      // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    .reg .b64 pair,scale_pair,bias_pair;                                    // Two-lane containers for scores, scale, and bias.
    mov.b64 scale_pair, 0x4100000041000000;                                 // Put the FP32 scale 8.0 into both lanes.
    mov.b64 bias_pair, {$9, $9};                                            // Put the supplied code bias into both lanes.
    mov.b64 pair, {$5, $6};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f0, f1}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    mov.b64 pair, {$7, $8};                                                 // Collect two input scores for one paired operation.
    fma.rn.f32x2 pair, pair, scale_pair, bias_pair;                         // Compute 8 * score + bias for two scores together.
    mov.b64 {f2, f3}, pair;                                                 // Expose the two mapped FP32 values as separate lanes.
    cvt.rni.s32.f32 c0, f0;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c1, f1;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c2, f2;                                                 // Round to a signed integer code; ties choose even.
    cvt.rni.s32.f32 c3, f3;                                                 // Round to a signed integer code; ties choose even.
    cvt.pack.sat.u8.s32.b32 packed, c3, c2, 0;                              // Saturate codes to 0..255 and place codes 2 and 3 in the low two bytes.
    cvt.pack.sat.u8.s32.b32 packed, c1, c0, packed;                         // Saturate codes 0 and 1, then pack all four bytes in score order.
    {                                                                       // Begin a private register scope.
        .reg .b32 native_source,half01,half23;                              // Saved packed bytes and the two converted FP16 pairs.
        .reg .b16 byte01,byte23,h0,h1,h2,h3;                                // Containers for FP8 byte pairs and individual FP16 values.
        mov.b32 native_source, packed;                                      // Save the packed input bytes before writing decoded outputs.
        mov.b32 {byte01,byte23}, native_source;                             // Split the packed word into lower and upper pairs of FP8 bytes.
        cvt.rn.f16x2.e4m3x2 half01, byte01;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        cvt.rn.f16x2.e4m3x2 half23, byte23;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        mov.b32 {h0,h1}, half01;                                            // Expose the lower and upper FP16 values separately.
        mov.b32 {h2,h3}, half23;                                            // Expose the lower and upper FP16 values separately.
        cvt.f32.f16 $1, h0;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $2, h1;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $3, h2;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $4, h3;                                                 // Widen one FP16 value exactly to the FP32 weight output.
    }                                                                       // End this private register scope.
    mov.b32 $0, packed;                                                     // Return the packed FP8 probability bytes for PV matmul.
}                                                                           // End this private register scope.
"""

# Decode only: byte extraction and FP32 bit construction.
_decode_assembly = """
{                                                                           // Begin a private register scope.
    .reg .b32 source,c,bits;                                                // Temporary integer codes, packed bytes, or FP32 bit patterns.
    .reg .pred sub;                                                         // Flag indicating that a code is zero or subnormal.
    mov.b32 source, $4;                                                     // Save the input bytes before output registers can overwrite the input.
    bfe.u32 c, source, 0, 8;                                                // Extract byte 0 as an unsigned integer code.
    add.u32 bits, c, 960;                                                   // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $0, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c, 8;                                                  // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $0, $0, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c, source, 8, 8;                                                // Extract byte 1 as an unsigned integer code.
    add.u32 bits, c, 960;                                                   // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $1, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c, 8;                                                  // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $1, $1, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c, source, 16, 8;                                               // Extract byte 2 as an unsigned integer code.
    add.u32 bits, c, 960;                                                   // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $2, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c, 8;                                                  // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $2, $2, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
    bfe.u32 c, source, 24, 8;                                               // Extract byte 3 as an unsigned integer code.
    add.u32 bits, c, 960;                                                   // Add 960 to change the E4M3 exponent bias into the FP32 exponent bias.
    shl.b32 bits, bits, 20;                                                 // Place the adjusted exponent and mantissa into FP32 bit positions.
    mov.b32 $3, bits;                                                       // Interpret the constructed bits as FP32 without numeric conversion.
    setp.lt.u32 sub, c, 8;                                                  // Check whether the code is zero or an E4M3 subnormal (code below 8).
    @sub fma.rn.f32 $3, $3, 0f40000000, 0fBC800000;                         // For zero/subnormals, compute 2 * value - 1/64 to obtain exactly code/512.
}                                                                           // End this private register scope.
"""

# Decode only: native FP8 -> FP16 -> FP32 conversion.
_native_decode_assembly = """
{                                                                           // Begin a private register scope.
    {                                                                       // Begin a private register scope.
        .reg .b32 native_source,half01,half23;                              // Saved packed bytes and the two converted FP16 pairs.
        .reg .b16 byte01,byte23,h0,h1,h2,h3;                                // Containers for FP8 byte pairs and individual FP16 values.
        mov.b32 native_source, $4;                                          // Save the packed input bytes before writing decoded outputs.
        mov.b32 {byte01,byte23}, native_source;                             // Split the packed word into lower and upper pairs of FP8 bytes.
        cvt.rn.f16x2.e4m3x2 half01, byte01;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        cvt.rn.f16x2.e4m3x2 half23, byte23;                                 // Convert two E4M3 bytes exactly into two packed FP16 values.
        mov.b32 {h0,h1}, half01;                                            // Expose the lower and upper FP16 values separately.
        mov.b32 {h2,h3}, half23;                                            // Expose the lower and upper FP16 values separately.
        cvt.f32.f16 $0, h0;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $1, h1;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $2, h2;                                                 // Widen one FP16 value exactly to the FP32 weight output.
        cvt.f32.f16 $3, h3;                                                 // Widen one FP16 value exactly to the FP32 weight output.
    }                                                                       // End this private register scope.
}                                                                           // End this private register scope.
"""

# Compile-time block selection and CuTe output bindings.

@cute.jit
def expcast_pack4(x0: Float32, x1: Float32, x2: Float32, x3: Float32,
                  bias: Float32, cap: Float32, *, remove_clamps: Constexpr[bool] = False,
                  packed_mapping: Constexpr[bool] = False,
                  packed_conversion: Constexpr[bool] = False,
                  loc=None, ip=None) -> Int32:
    # x already includes the P log2 offset. Unsigned conversion maps negative
    # values and masked -inf to zero. remove_clamps requires the caller to
    # prove the upper code bound. Codes are bits, not numeric FP8 values.
    selected = _unclamped_assembly if const_expr(remove_clamps) else assembly
    if const_expr(packed_mapping):
        selected = _packed_unclamped_assembly if const_expr(remove_clamps) else _packed_assembly
    if const_expr(packed_conversion):
        selected = _unclamped_pack_conversion_assembly if const_expr(remove_clamps) else _pack_conversion_assembly
        if const_expr(packed_mapping):
            selected = _paired_unclamped_pack_conversion_assembly if const_expr(remove_clamps) else _paired_pack_conversion_assembly
    result = llvm.inline_asm(
        T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0,x1,x2,x3,bias,cap)],
        selected,
        "=r,f,f,f,f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(result)

@cute.jit
def expcast_pack_decode4(
    x0: Float32, x1: Float32, x2: Float32, x3: Float32,
    bias: Float32, cap: Float32, *, remove_clamps: Constexpr[bool] = False,
    packed_mapping: Constexpr[bool] = False,
    packed_conversion: Constexpr[bool] = True,
    native_decode: Constexpr[bool] = False,
    loc=None, ip=None,
) -> tuple[Int32, Float32, Float32, Float32, Float32]:
    """Encode E4M3 P and decode its exact FP32 weights.

    packed_conversion=True selects cvt.pack plus byte extraction for decoding;
    False retains the previous independent conversion/packing/decoding path.
    native_decode=True uses exact E4M3-to-FP16-to-FP32 hardware conversion.
    All inputs are consumed before writing outputs, allowing register reuse.
    Codes 0..7 include zero/subnormals; 8..126 include all positive normals.
    remove_clamps is only valid when the caller proves the upper code bound.
    """
    # Constexpr selection emits only the selected PTX string, not both paths.
    selected = _unclamped_fused_assembly if const_expr(remove_clamps) else _fused_assembly
    if const_expr(packed_mapping):
        selected = _packed_unclamped_fused_assembly if const_expr(remove_clamps) else _packed_fused_assembly
    if const_expr(packed_conversion):
        selected = _unclamped_pack_conversion_fused_assembly if const_expr(remove_clamps) else _pack_conversion_fused_assembly
        if const_expr(packed_mapping):
            selected = _packed_unclamped_pack_conversion_fused_assembly if const_expr(remove_clamps) else _packed_pack_conversion_fused_assembly
    if const_expr(native_decode):
        selected = _native_unclamped_fused if const_expr(remove_clamps) else _native_fused
        if const_expr(packed_mapping):
            selected = _native_paired_unclamped_fused if const_expr(remove_clamps) else _native_paired_fused
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32(), T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3, bias, cap)],
        selected,
        "=r,=f,=f,=f,=f,f,f,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [4], loc=loc, ip=ip)),
    )

@cute.jit
def expcast_decode4(packed: Int32, *, native_decode: Constexpr[bool] = False, loc=None, ip=None):
    """Exact decoding for diagnostics and publication-before-denominator paths."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(packed).ir_value(loc=loc, ip=ip)],
        _native_decode_assembly if const_expr(native_decode) else _decode_assembly, "=f,=f,=f,=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )
