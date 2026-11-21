// Encode a real descriptor producer and consumer into one Metal command buffer.
// The test passes retained buffers from the same MLX allocator the engine replays on.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int encode_indirect(void* raw_count, void* raw_grid, uint64_t grid_offset,
                    void* raw_result, bool external_grid) {
  @autoreleasepool {
    id<MTLBuffer> count = (__bridge id<MTLBuffer>)raw_count;
    id<MTLBuffer> grid = (__bridge id<MTLBuffer>)raw_grid;
    id<MTLBuffer> result = (__bridge id<MTLBuffer>)raw_result;
    id<MTLDevice> device = result.device;
    NSString* source = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void prepare_test_grid(device const uint* n [[buffer(0)]], device uint* grid [[buffer(1)]], "
        "device uint* result [[buffer(2)]], uint i [[thread_position_in_grid]]) {"
        "result[i] = 0; if (i == 0) { grid[0] = min(n[0], 16u); grid[1] = 1; grid[2] = 1; } }"
        "kernel void clear_test_result(device uint* result [[buffer(0)]], uint i [[thread_position_in_grid]]) { result[i] = 0; }"
        "kernel void fill_test_result(device uint* result [[buffer(0)]], uint i [[thread_position_in_grid]]) { result[i] = i + 1; }";
    NSError* error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
    id<MTLComputePipelineState> first = [device newComputePipelineStateWithFunction:
        [lib newFunctionWithName:(external_grid ? @"clear_test_result" : @"prepare_test_grid")] error:&error];
    id<MTLComputePipelineState> second = [device newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"fill_test_result"] error:&error];
    if (!first || !second) return 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [enc setComputePipelineState:first];
    if (external_grid) {
      [enc setBuffer:result offset:0 atIndex:0];
    } else {
      [enc setBuffer:count offset:0 atIndex:0];
      [enc setBuffer:grid offset:grid_offset atIndex:1];
      [enc setBuffer:result offset:0 atIndex:2];
    }
    [enc dispatchThreads:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(16,1,1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:second];
    [enc setBuffer:result offset:0 atIndex:0];
    [enc dispatchThreadgroupsWithIndirectBuffer:grid indirectBufferOffset:grid_offset threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [enc endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    return buffer.status == MTLCommandBufferStatusCompleted ? 0 : 2;
  }
}
