# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Streaming Video Dataset Registration

This module registers streaming video datasets for swift training.
"""

from ..preprocessor import StreamingVideoPreprocessor, StreamingVideoMessagesPreprocessor
from ..register import DatasetMeta, register_dataset

# Re-export for convenience
__all__ = [
    'StreamingVideoPreprocessor',
    'StreamingVideoMessagesPreprocessor',
    'register_streaming_video_dataset',
]


def register_streaming_video_dataset(
    dataset_path: str,
    dataset_name: str = 'streaming_video',
    fps: float = 1.0,
    max_frames: int = None,
    frame_output_dir: str = None,
    save_frames: bool = False,
    num_workers: int = 8,  # Number of threads for parallel frame saving
    enable_memory_cache: bool = True,  # Enable in-memory LRU cache
    **kwargs
) -> DatasetMeta:
    """
    Register a streaming video dataset.
    
    Args:
        dataset_path: Path to JSON data file
        dataset_name: Name for the dataset
        fps: Frames per second to extract
        max_frames: Maximum frames per video
        frame_output_dir: Directory to save extracted frames (only used if save_frames=True)
        save_frames: Whether to save frames to disk (False = in-memory PIL images)
        num_workers: Number of threads for parallel frame saving
        enable_memory_cache: Enable in-memory LRU cache for repeated access (default: True)
        **kwargs: Additional DatasetMeta arguments
        
    Returns:
        Registered DatasetMeta
        
    Example:
        register_streaming_video_dataset(
            dataset_path='/path/to/data.json',
            dataset_name='my_streaming_video',
            fps=1.0,
            save_frames=True,  # Enable disk caching
            frame_output_dir='./frames',
            enable_memory_cache=True,  # Enable memory cache
        )
    """
    preprocessor = StreamingVideoPreprocessor(
        fps=fps,
        max_frames=max_frames,
        frame_output_dir=frame_output_dir,
        save_frames=save_frames,
        num_workers=num_workers,
        enable_memory_cache=enable_memory_cache,
    )
    
    meta = DatasetMeta(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        preprocess_func=preprocessor,
        tags=['streaming-video', 'multi-modal', 'video'],
        **kwargs
    )
    
    register_dataset(meta)
    return meta
