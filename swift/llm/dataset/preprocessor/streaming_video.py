# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Streaming Video Data Preprocessor

This module handles the <stream> token for streaming video analysis training.
When a <stream> token is detected, the video is automatically extracted to frames
and the messages are expanded to multi-round conversations with <image> tokens.

Input format (using <stream> token):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Find all events<stream>"},
        {"role": "assistant", "content": "</Standby>"},
        {"role": "user", "content": "<stream>"},
        {"role": "assistant", "content": "</Standby>"},
        ...
    ],
    "videos": ["/path/to/video.mp4"]
}

Output format (converted):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Find all events<image>"},
        {"role": "assistant", "content": "</Standby>"},
        {"role": "user", "content": "<image>"},
        {"role": "assistant", "content": "</Standby>"},
        ...
    ],
    "images": [PIL.Image, PIL.Image, ...]  # In-memory PIL images or file paths
}

The <stream> token is replaced with <image> token, and video is extracted to frames.
"""

import fcntl
import json
import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from swift.utils import get_logger
from .core import RowPreprocessor, MessagesPreprocessor

logger = get_logger()

# Special token for streaming video (to distinguish from normal video tasks)
STREAM_TOKEN = '<stream>'
IMAGE_TOKEN = '<image>'

# Cache configuration
_MAX_CACHE_SIZE = 1000  # Maximum number of videos to cache
_MAX_CACHE_MEMORY_MB = 262144  # Maximum memory for cache in MB (256GB default)


class LRUFrameCache:
    """
    Thread-safe LRU cache for video frames with memory size limit.
    
    Uses OrderedDict for true LRU behavior - moves accessed items to end.
    Tracks both entry count and memory usage for eviction.
    """
    
    def __init__(self, max_entries: int = _MAX_CACHE_SIZE, max_memory_mb: int = _MAX_CACHE_MEMORY_MB):
        self._cache: OrderedDict[str, List[Image.Image]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory_bytes = 0
    
    @staticmethod
    def _estimate_image_size(img: Image.Image) -> int:
        """Estimate memory size of a PIL Image in bytes."""
        # width * height * channels (assume 3 for RGB)
        return img.width * img.height * 3
    
    def _estimate_frames_size(self, frames: List[Image.Image]) -> int:
        """Estimate total memory size of frames list."""
        return sum(self._estimate_image_size(f) for f in frames)
    
    def get(self, key: str) -> Optional[List[Image.Image]]:
        """Get frames from cache, moving to end for LRU."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                logger.debug(f"Cache hit for {key}")
                return self._cache[key]
        return None
    
    def put(self, key: str, frames: List[Image.Image]) -> None:
        """Store frames in cache with LRU eviction."""
        frames_size = self._estimate_frames_size(frames)
        
        with self._lock:
            # If key already exists, remove old entry first
            if key in self._cache:
                old_frames = self._cache.pop(key)
                self._current_memory_bytes -= self._estimate_frames_size(old_frames)
            
            # Evict entries until we have enough space
            while (
                (len(self._cache) >= self._max_entries or 
                 self._current_memory_bytes + frames_size > self._max_memory_bytes)
                and self._cache
            ):
                # Remove oldest (first) entry
                oldest_key, oldest_frames = self._cache.popitem(last=False)
                self._current_memory_bytes -= self._estimate_frames_size(oldest_frames)
                logger.debug(f"Cache evicted: {oldest_key}")
            
            # Add new entry
            self._cache[key] = frames
            self._current_memory_bytes += frames_size
            logger.debug(f"Cache stored: {key} ({len(frames)} frames, {frames_size / 1024 / 1024:.2f} MB)")
    
    def clear(self) -> int:
        """Clear cache and return number of entries cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_memory_bytes = 0
            return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cached_videos": len(self._cache),
                "total_frames": sum(len(frames) for frames in self._cache.values()),
                "memory_usage_mb": self._current_memory_bytes / 1024 / 1024,
                "max_entries": self._max_entries,
                "max_memory_mb": self._max_memory_bytes / 1024 / 1024,
            }


# Global frame cache instance
_frame_cache = LRUFrameCache()


class FileLock:
    """
    Simple file-based lock for cross-process synchronization.
    
    Uses fcntl.flock on Unix systems for atomic locking.
    Falls back to a simple marker file on Windows.
    """
    
    def __init__(self, lock_path: str, timeout: float = 60.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._lock_file = None
    
    def __enter__(self):
        # Create lock file directory if needed
        lock_dir = os.path.dirname(self.lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        
        self._lock_file = open(self.lock_path, 'w')
        
        if sys.platform != 'win32':
            # Unix: use fcntl for proper file locking
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        else:
            # Windows: simple approach - just use the file existence
            # Note: This is not perfectly atomic but better than nothing
            pass
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_file:
            if sys.platform != 'win32':
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
        return False


def _save_frame_worker(args: Tuple[str, np.ndarray]) -> str:
    """Worker function to save a single frame to disk."""
    frame_path, frame = args
    # Write to temp file first, then rename for atomicity
    temp_path = frame_path + '.tmp'
    cv2.imwrite(temp_path, frame)
    os.rename(temp_path, frame_path)
    return frame_path


class VideoFrameExtractor:
    """Utility class for extracting frames from video with optimized seek-based extraction."""
    
    def __init__(
        self,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
        use_cache: bool = True,
        save_frames: bool = False,
        num_workers: int = 8,  # Number of threads for parallel frame saving
        enable_memory_cache: bool = True,  # Enable in-memory caching for repeated access
        frame_tolerance: int = 1,  # Tolerance in frames for count mismatch (at current fps)
    ):
        self.fps = fps
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.save_frames = save_frames
        self.num_workers = num_workers
        self.enable_memory_cache = enable_memory_cache
        self.frame_tolerance = frame_tolerance  # 1 frame = 1 second at 1fps
    
    def _get_cache_key(self, video_path: str) -> str:
        """Generate cache key based on video path and extraction parameters."""
        return f"{video_path}|fps={self.fps}|max={self.max_frames}"
    
    def _get_from_cache(self, video_path: str) -> Optional[List[Image.Image]]:
        """Try to get frames from memory cache using LRU cache."""
        if not self.enable_memory_cache:
            return None
        return _frame_cache.get(self._get_cache_key(video_path))
    
    def _put_to_cache(self, video_path: str, frames: List[Image.Image]) -> None:
        """Store frames in memory cache with LRU eviction."""
        if not self.enable_memory_cache:
            return
        _frame_cache.put(self._get_cache_key(video_path), frames)
    
    def extract(self, video_path: str) -> List[Union[str, Image.Image]]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of PIL Images (in-memory) or file paths (if save_frames=True)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # If saving frames, use file-based approach
        if self.save_frames:
            return self._extract_to_files(video_path)
        
        # Check memory cache first (for in-memory mode)
        cached_frames = self._get_from_cache(video_path)
        if cached_frames is not None:
            return cached_frames
        
        # Extract to memory with optimized seek
        frames = self._extract_to_memory_optimized(video_path)
        
        # Store in cache
        self._put_to_cache(video_path, frames)
        
        return frames
    
    def _extract_to_memory_optimized(self, video_path: str) -> List[Image.Image]:
        """
        Extract frames directly to memory using optimized seek-based approach.
        
        Instead of reading all frames sequentially and skipping unwanted ones,
        this method directly seeks to target frame positions, which is much faster
        for low fps extraction (e.g., 1 fps from 30 fps video).
        
        For better seek accuracy on compressed videos (H.264), we seek to slightly
        before the target and read forward to ensure we get the right frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Handle invalid video metadata
            if video_fps <= 0 or total_frames <= 0:
                logger.warning(f"Invalid video metadata for {video_path}, falling back to sequential read")
                return self._extract_to_memory_sequential(cap)
            
            # Calculate frame interval
            frame_interval = max(1, int(video_fps / self.fps)) if self.fps > 0 else 1
            
            frames = []
            target_frame = 0
            
            # For accurate seeking on compressed videos, we use a hybrid approach:
            # 1. Try direct seek first
            # 2. If seek seems inaccurate (position doesn't match), fall back to sequential
            seek_accurate = True
            
            while target_frame < total_frames:
                if seek_accurate:
                    # Try to seek directly
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    
                    # Verify seek accuracy by checking position
                    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if abs(actual_pos - target_frame) > frame_interval // 2:
                        # Seek is inaccurate, fall back to sequential for remaining frames
                        logger.debug(f"Seek inaccurate at frame {target_frame} (got {actual_pos}), switching to sequential")
                        seek_accurate = False
                        # Reset to beginning and skip to current position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        for _ in range(target_frame):
                            cap.read()
                
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB and create PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
                if self.max_frames and len(frames) >= self.max_frames:
                    break
                
                if seek_accurate:
                    target_frame += frame_interval
                else:
                    # Sequential mode: skip frames
                    for _ in range(frame_interval - 1):
                        ret, _ = cap.read()
                        if not ret:
                            break
                    target_frame += frame_interval
            
            return frames
            
        finally:
            cap.release()
    
    def _extract_to_memory_sequential(self, cap: cv2.VideoCapture) -> List[Image.Image]:
        """Fallback sequential extraction when seek is not reliable."""
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.fps)) if self.fps > 0 and video_fps > 0 else 1
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
                if self.max_frames and len(frames) >= self.max_frames:
                    break
            
            frame_idx += 1
        
        return frames
    
    def _extract_to_files(self, video_path: str) -> List[str]:
        """Extract frames to disk files using optimized seek and multi-threading.
        
        Uses file locking to prevent race conditions when multiple processes
        try to extract frames from the same video simultaneously.
        """
        # Determine output directory
        if self.output_dir:
            output_dir = self.output_dir
        else:
            output_dir = os.path.dirname(video_path)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(output_dir, f"{video_name}_frames")
        done_marker = os.path.join(frame_dir, ".done")
        lock_path = os.path.join(output_dir, f".{video_name}_frames.lock")
        
        # Check if already processed (use .done marker file) - quick check without lock
        if self.use_cache and os.path.exists(done_marker):
            existing = sorted([
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            if existing:
                logger.debug(f"Skipping already processed video: {video_path}")
                if self.max_frames:
                    return existing[:self.max_frames]
                return existing
        
        # Use file lock to prevent race conditions
        with FileLock(lock_path):
            # Double-check after acquiring lock (another process may have finished)
            if self.use_cache and os.path.exists(done_marker):
                existing = sorted([
                    os.path.join(frame_dir, f)
                    for f in os.listdir(frame_dir)
                    if f.endswith(('.png', '.jpg', '.jpeg'))
                ])
                if existing:
                    logger.debug(f"Skipping already processed video (after lock): {video_path}")
                    if self.max_frames:
                        return existing[:self.max_frames]
                    return existing
            
            os.makedirs(frame_dir, exist_ok=True)
            
            # Extract frames from video using optimized seek
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            try:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_interval = max(1, int(video_fps / self.fps)) if self.fps > 0 else 1
                
                # Collect frames to save
                frames_to_save = []  # List of (frame_path, frame_data)
                frame_paths = []
                saved_idx = 0
                
                # Use optimized seek if metadata is valid
                use_seek = video_fps > 0 and total_frames > 0
                
                if use_seek:
                    # Optimized: seek directly to target frames with accuracy check
                    target_frame = 0
                    seek_accurate = True
                    sequential_frame_idx = 0  # Counter for sequential mode
                    
                    while target_frame < total_frames:
                        if seek_accurate:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                            actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            if abs(actual_pos - target_frame) > frame_interval // 2:
                                logger.debug(f"Seek inaccurate at frame {target_frame}, switching to sequential")
                                seek_accurate = False
                                # Reset and read sequentially to target position
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                sequential_frame_idx = 0
                                # Skip to current target frame
                                while sequential_frame_idx < target_frame:
                                    ret, _ = cap.read()
                                    if not ret:
                                        break
                                    sequential_frame_idx += 1
                        
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if seek_accurate:
                            # In seek mode, we directly jumped to target_frame
                            current_frame = target_frame
                        else:
                            # In sequential mode, track current position
                            current_frame = sequential_frame_idx
                            sequential_frame_idx += 1
                        
                        frame_path = os.path.join(frame_dir, f"frame_{saved_idx:06d}.png")
                        frame_paths.append(frame_path)
                        
                        if not os.path.exists(frame_path):
                            frames_to_save.append((frame_path, frame.copy()))
                        
                        saved_idx += 1
                        
                        if self.max_frames and saved_idx >= self.max_frames:
                            break
                        
                        # Calculate next target frame
                        if seek_accurate:
                            target_frame += frame_interval
                        else:
                            # In sequential mode, skip frames to reach next target
                            next_target = current_frame + frame_interval
                            while sequential_frame_idx < next_target:
                                ret, _ = cap.read()
                                if not ret:
                                    break
                                sequential_frame_idx += 1
                            target_frame = next_target
                else:
                    # Fallback: sequential read when metadata is invalid
                    frame_idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_idx % frame_interval == 0:
                            frame_path = os.path.join(frame_dir, f"frame_{saved_idx:06d}.png")
                            frame_paths.append(frame_path)
                            
                            if not os.path.exists(frame_path):
                                frames_to_save.append((frame_path, frame.copy()))
                            
                            saved_idx += 1
                            
                            if self.max_frames and saved_idx >= self.max_frames:
                                break
                        
                        frame_idx += 1
            finally:
                cap.release()
            
            # Save frames using multi-threading, track failures
            failed_frames = []
            if frames_to_save:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_path = {
                        executor.submit(_save_frame_worker, args): args[0] 
                        for args in frames_to_save
                    }
                    for future in as_completed(future_to_path):
                        frame_path = future_to_path[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.warning(f"Failed to save frame {frame_path}: {e}")
                            failed_frames.append(frame_path)
            
            # Only create .done marker if all frames saved successfully
            if failed_frames:
                logger.error(
                    f"Failed to save {len(failed_frames)} frames from {video_path}, "
                    f"not creating .done marker"
                )
            else:
                with open(done_marker, 'w') as f:
                    f.write(f"frames={len(frame_paths)}\n")
                    f.write(f"video_fps={video_fps}\n")
                    f.write(f"extraction_fps={self.fps}\n")
            
            logger.info(
                f"Extracted {len(frame_paths)} frames from {video_path} "
                f"(saved {len(frames_to_save) - len(failed_frames)} new frames)"
            )
        
        return frame_paths
    
    def get_frame_count(self, video_path: str) -> int:
        """Get estimated frame count at current fps."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_interval = max(1, int(video_fps / self.fps)) if self.fps > 0 else 1
        estimated = total_frames // frame_interval
        
        if self.max_frames:
            return min(estimated, self.max_frames)
        return estimated


class StreamingVideoPreprocessor(RowPreprocessor):
    """
    Preprocessor that handles <stream> token for streaming video data.
    
    When <stream> token is detected in messages, the video is extracted to frames
    and <stream> tokens are replaced with <image> tokens.
    
    Input format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Find all events<stream>"},
            {"role": "assistant", "content": "</Standby>"},
            {"role": "user", "content": "<stream>"},
            {"role": "assistant", "content": "</Response> ..."},
            ...
        ],
        "videos": ["/path/to/video.mp4"]
    }
    
    Output format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Find all events<image>"},
            {"role": "assistant", "content": "</Standby>"},
            {"role": "user", "content": "<image>"},
            {"role": "assistant", "content": "</Response> ..."},
            ...
        ],
        "images": [PIL.Image, ...]  # In-memory PIL images
    }
    """
    
    def __init__(
        self,
        *,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
        frame_output_dir: Optional[str] = None,
        use_frame_cache: bool = True,
        save_frames: bool = False,
        num_workers: int = 8,  # Number of threads for parallel frame saving
        enable_memory_cache: bool = True,  # Enable in-memory caching for multi-epoch training
        frame_tolerance: int = 1,  # Tolerance in frames for count mismatch (default: 1 second at 1fps)
        columns: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Args:
            fps: Frames per second to extract from video
            max_frames: Maximum number of frames per video
            frame_output_dir: Directory to save extracted frames (only used if save_frames=True)
            use_frame_cache: Whether to use cached frames if available (only used if save_frames=True)
            save_frames: Whether to save frames to disk (False = in-memory PIL images)
            num_workers: Number of threads for parallel frame saving
            enable_memory_cache: Enable in-memory caching to avoid re-decoding videos (for multi-epoch)
            frame_tolerance: Allowed difference between extracted frames and expected tokens (default: 1)
            columns: Column mapping
        """
        super().__init__(columns=columns, **kwargs)
        self.fps = fps
        self.max_frames = max_frames
        self.frame_output_dir = frame_output_dir
        self.save_frames = save_frames
        self.num_workers = num_workers
        self.enable_memory_cache = enable_memory_cache
        self.frame_tolerance = frame_tolerance
        
        self.frame_extractor = VideoFrameExtractor(
            fps=fps,
            max_frames=max_frames,
            output_dir=frame_output_dir,
            use_cache=use_frame_cache,
            save_frames=save_frames,
            num_workers=num_workers,
            enable_memory_cache=enable_memory_cache,
            frame_tolerance=frame_tolerance,
        )
    
    def _has_stream_token(self, row: Dict[str, Any]) -> bool:
        """Check if row contains <stream> token."""
        messages = row.get('messages', [])
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str) and STREAM_TOKEN in content:
                return True
        return False
    
    def _count_stream_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count total <stream> tokens in messages."""
        count = 0
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str):
                count += content.count(STREAM_TOKEN)
        return count
    
    def _get_video_path(self, row: Dict[str, Any]) -> Optional[str]:
        """Extract video path from row."""
        # videos list (primary)
        videos = row.get('videos', [])
        if videos:
            if isinstance(videos, str):
                return videos
            elif isinstance(videos, list) and videos:
                return videos[0]
        
        # video field
        if 'video' in row and row['video']:
            video = row['video']
            if isinstance(video, str):
                return video
            elif isinstance(video, list) and video:
                return video[0]
        
        # video_path field (fallback)
        if 'video_path' in row and row['video_path']:
            return row['video_path']
        
        return None
    
    def _replace_stream_with_image(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Replace <stream> tokens with <image> tokens in messages."""
        new_messages = []
        for msg in messages:
            new_msg = msg.copy()
            content = msg.get('content', '')
            if isinstance(content, str) and STREAM_TOKEN in content:
                new_msg['content'] = content.replace(STREAM_TOKEN, IMAGE_TOKEN)
            new_messages.append(new_msg)
        return new_messages
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Preprocess a single row.
        
        If <stream> token is detected:
        1. Extract frames from video
        2. Replace <stream> with <image> in messages
        3. Add images list
        
        Frame count tolerance:
        - If frames differ from token count by <= frame_tolerance, auto-adjust
        - If more frames: truncate to match token count
        - If fewer frames: duplicate last frame to match token count
        """
        try:
            # Check if this is streaming video data
            if not self._has_stream_token(row):
                # Not streaming data, return as-is
                return row
            
            # Get video path
            video_path = self._get_video_path(row)
            if not video_path:
                logger.warning(f"No video path found in row")
                return None
            
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}")
                return None
            
            # Count stream tokens to determine expected frame count
            messages = row.get('messages', [])
            stream_count = self._count_stream_tokens(messages)
            
            # Extract frames from video
            frame_paths = self.frame_extractor.extract(video_path)
            if not frame_paths:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            # Validate frame count with tolerance
            frame_diff = len(frame_paths) - stream_count
            
            if abs(frame_diff) > self.frame_tolerance:
                logger.warning(
                    f"Frame count mismatch: extracted {len(frame_paths)} frames but "
                    f"found {stream_count} <stream> tokens in {video_path}. "
                    f"Difference ({abs(frame_diff)}) exceeds tolerance ({self.frame_tolerance}). Discarding sample."
                )
                return None
            
            # Auto-adjust frame count to match token count
            if frame_diff != 0:
                if frame_diff > 0:
                    # More frames than tokens: truncate
                    logger.debug(f"Truncating {frame_diff} extra frames from {video_path}")
                    frame_paths = frame_paths[:stream_count]
                else:
                    # Fewer frames than tokens: duplicate last frame
                    logger.debug(f"Duplicating last frame {-frame_diff} times for {video_path}")
                    last_frame = frame_paths[-1]
                    frame_paths = frame_paths + [last_frame] * (-frame_diff)
            
            # Replace <stream> with <image> in messages
            new_messages = self._replace_stream_with_image(messages)
            
            # Build result
            result = {
                'messages': new_messages,
                'images': frame_paths,
            }
            
            # Preserve other fields (except videos since we converted to images)
            for key, value in row.items():
                if key not in ('messages', 'videos', 'video', 'video_path', 'images'):
                    result[key] = value
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to preprocess streaming video: {e}")
            return None


class StreamingVideoMessagesPreprocessor(MessagesPreprocessor):
    """
    Extended MessagesPreprocessor that handles <stream> token.
    
    This preprocessor first checks for <stream> token. If found, it extracts
    video frames and replaces <stream> with <image>. Then it applies standard
    messages preprocessing.
    """
    
    def __init__(
        self,
        *,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
        frame_output_dir: Optional[str] = None,
        use_frame_cache: bool = True,
        save_frames: bool = False,
        num_workers: int = 8,  # Number of threads for parallel frame saving
        enable_memory_cache: bool = True,  # Enable in-memory caching for multi-epoch training
        frame_tolerance: int = 1,  # Tolerance in frames for count mismatch
        # MessagesPreprocessor args
        role_key: Optional[str] = None,
        content_key: Optional[str] = None,
        user_role: Optional[str] = None,
        assistant_role: Optional[str] = None,
        system_role: str = 'system',
        columns: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(
            role_key=role_key,
            content_key=content_key,
            user_role=user_role,
            assistant_role=assistant_role,
            system_role=system_role,
            columns=columns,
            **kwargs
        )
        
        # Streaming video preprocessor
        self.stream_preprocessor = StreamingVideoPreprocessor(
            fps=fps,
            max_frames=max_frames,
            frame_output_dir=frame_output_dir,
            use_frame_cache=use_frame_cache,
            save_frames=save_frames,
            num_workers=num_workers,
            enable_memory_cache=enable_memory_cache,
            frame_tolerance=frame_tolerance,
        )
    
    def _has_stream_token(self, row: Dict[str, Any]) -> bool:
        """Check if row contains <stream> token."""
        messages = row.get('messages', row.get('conversations', []))
        for msg in messages:
            content = msg.get('content', msg.get('value', ''))
            if isinstance(content, str) and STREAM_TOKEN in content:
                return True
        return False
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Preprocess row, handling <stream> token specially.
        """
        if self._has_stream_token(row):
            # First convert streaming video format
            row = self.stream_preprocessor.preprocess(row)
            if row is None:
                return None
        
        # Then apply standard messages preprocessing
        return super().preprocess(row)


# Convenience function to clear frame cache
def clear_frame_cache():
    """Clear the in-memory frame cache to free memory."""
    count = _frame_cache.clear()
    logger.info(f"Cleared frame cache ({count} entries)")


def get_frame_cache_stats() -> Dict[str, Any]:
    """Get frame cache statistics."""
    return _frame_cache.stats()


# Convenience function for direct conversion
def convert_streaming_video_data(
    input_path: str,
    output_path: str,
    fps: float = 1.0,
    output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert streaming video data from <stream> format to <image> format.
    
    Args:
        input_path: Input JSON file path
        output_path: Output JSON file path
        fps: Frames per second to extract
        output_dir: Directory to save frames
        max_frames: Maximum frames per video
        
    Returns:
        List of converted samples
    """
    from tqdm import tqdm
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Handle both single object and list
        data = json.loads(content)
        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data
    
    preprocessor = StreamingVideoPreprocessor(
        fps=fps,
        max_frames=max_frames,
        frame_output_dir=output_dir,
    )
    
    results = []
    for item in tqdm(data_list, desc="Converting"):
        try:
            result = preprocessor.preprocess(item)
            if result:
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to convert: {e}")
    
    # Output single object if input was single object
    output_data = results[0] if len(results) == 1 else results
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(results)} samples to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert streaming video data")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS for extraction")
    parser.add_argument("--output-dir", help="Frame output directory")
    parser.add_argument("--max-frames", type=int, help="Max frames per video")
    
    args = parser.parse_args()
    
    convert_streaming_video_data(
        input_path=args.input,
        output_path=args.output,
        fps=args.fps,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
