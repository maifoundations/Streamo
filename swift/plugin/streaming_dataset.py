# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Register streaming video dataset with preprocessor.

This plugin registers the streaming video dataset so that:
1. <stream> tokens are replaced with <image> tokens
2. Videos are extracted to frames (with disk caching for efficiency)
3. Data is processed as multi-image input (not video)

Performance optimizations:
- save_frames=True: Frames are cached on disk, avoiding repeated video decoding
- enable_memory_cache=True: LRU cache for in-memory frames in current epoch
- Optimized seek-based frame extraction (only reads target frames, not all frames)
"""

from swift.llm.dataset.dataset import register_streaming_video_dataset

# Register the streaming video dataset with optimized settings
# save_frames=True enables disk caching - first run extracts frames, subsequent runs load from disk
register_streaming_video_dataset(
    dataset_path='./dataset/stream/llava.jsonl',
    dataset_name='streaming_video',
    fps=1.0,
    max_frames=None,  # No limit
    save_frames=True,  # Enable disk caching for efficiency (recommended for multi-epoch training)
    frame_output_dir='./dataset/stream/frames',  # Frame cache directory
    enable_memory_cache=True,  # Enable in-memory LRU cache
)

print("Registered streaming_video dataset with optimized frame extraction (disk cache enabled)")
