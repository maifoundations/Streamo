#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert streaming video data between formats.

Supports two conversion modes:
1. to-stream: Convert format.json to videos+<stream> multi-round format
2. to-image: Convert videos+<stream> format to images+<image> format (with frame extraction)

Usage:
    # Convert from format.json to stream format (multi-round with <stream> tokens)
    python scripts/convert_streaming_video.py to-stream \
        --input format.json \
        --output stream_format.json \
        --fps 1.0

    # Convert from stream format to image format (extract frames, replace <stream> with <image>)
    python scripts/convert_streaming_video.py to-image \
        --input stream_format.json \
        --output converted_output.json \
        --fps 1.0 \
        --frame-dir /path/to/frames

Input format (format.json):
[
    {
        "video_name": "video1",
        "video_path": "/path/to/video.mp4",
        "task_type": "event_detection",
        "source": "custom",
        "question": [
            {"content": "Find all significant events", "time": "0"}
        ],
        "response": [
            {"content": "A man walks in", "st_time": "0", "end_time": "5"},
            {"content": "He picks up a book", "time": "10"}
        ]
    }
]

Stream format output (with <stream> tokens):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Question<stream>"},
        {"role": "assistant", "content": "</Standby>"},
        {"role": "user", "content": "<stream>"},
        {"role": "assistant", "content": "</Response> ..."},
        ...
    ],
    "videos": ["/path/to/video.mp4"]
}

Image format output (with <image> tokens):
{
    "messages": [...],  // same structure, but <stream> -> <image>
    "images": ["/path/to/frame_0.png", ...]
}
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


IMAGE_TOKEN = '<image>'
STREAM_TOKEN = '<stream>'

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant specializing in streaming video analysis. 
You will receive input frame by frame, each labeled with absolute time intervals 
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.  

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.  

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.  

Do not provide partial answers or speculate beyond the given information.  
Whenever you deliver an answer, begin with </Response>.
"""

# Response states
STATE_SILENCE = "</Silence>"
STATE_STANDBY = "</Standby>"
STATE_RESPONSE = "</Response>"


def get_video_frame_count(video_path: str, fps: float = 1.0) -> int:
    """Get estimated frame count at specified fps."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    frame_interval = max(1, int(video_fps / fps)) if fps > 0 else 1
    return total_frames // frame_interval


def parse_time(time_str: Union[str, int, float, None], fps: float = 1.0) -> Optional[int]:
    """Parse time string to frame index."""
    if time_str is None or time_str == '':
        return None
    
    if isinstance(time_str, (int, float)):
        return int(time_str)
    
    time_str = str(time_str).strip()
    if not time_str:
        return None
    
    # Try integer (frame index)
    try:
        return int(time_str)
    except ValueError:
        pass
    
    # Try float (seconds -> frame index)
    try:
        return int(float(time_str) * fps)
    except ValueError:
        pass
    
    # Try timestamp (HH:MM:SS or MM:SS)
    parts = time_str.split(':')
    try:
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return int((minutes * 60 + seconds) * fps)
        elif len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return int((hours * 3600 + minutes * 60 + seconds) * fps)
    except ValueError:
        pass
    
    return None


def build_frame_states(
    num_frames: int,
    questions: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    fps: float = 1.0,
) -> Tuple[List[str], Dict[int, str]]:
    """
    Build frame states based on questions and responses.
    
    Returns:
        (frame_states, question_map)
    """
    # Initialize all frames as silence
    frame_states = [STATE_SILENCE] * num_frames
    question_map = {}
    
    # Process questions
    # time=5 means the question appears at <4s-5s>, which is frame index 4 (time-1)
    for q in questions:
        content = q.get('content', '')
        time = parse_time(q.get('time'), fps)
        if content and time is not None:
            # time=5 -> frame index 4 (corresponds to <4s-5s>)
            frame_idx = max(0, time - 1) if time > 0 else 0
            if 0 <= frame_idx < num_frames:
                question_map[frame_idx] = content
    
    # Process responses
    # time values represent seconds, frame index = time - 1
    # e.g., st_time=5, end_time=7 means standby at frames 4,5,6 (<4s-5s>, <5s-6s>, <6s-7s>)
    #       response at frame 7 (<7s-8s>)
    for resp in responses:
        content = resp.get('content', '')
        st_time = parse_time(resp.get('st_time'), fps)
        end_time = parse_time(resp.get('end_time'), fps)
        time = parse_time(resp.get('time'), fps)
        
        if st_time is not None and end_time is not None:
            # Convert to frame indices: time=5 -> frame 4
            st_frame = max(0, st_time - 1) if st_time > 0 else 0
            end_frame = max(0, end_time - 1) if end_time > 0 else 0
            
            # Standby during [st_frame, end_frame]
            for i in range(st_frame, min(end_frame + 1, num_frames)):
                frame_states[i] = STATE_STANDBY
            
            # Response at end_frame + 1
            resp_idx = end_frame + 1
            if 0 <= resp_idx < num_frames and content:
                frame_states[resp_idx] = f"{STATE_RESPONSE} {content}"
                
        elif time is not None:
            # Direct response: time=8 -> frame 7 (<7s-8s>)
            frame_idx = max(0, time - 1) if time > 0 else 0
            if 0 <= frame_idx < num_frames and content:
                frame_states[frame_idx] = f"{STATE_RESPONSE} {content}"
    
    return frame_states, question_map


def convert_format_to_stream(
    data: Dict[str, Any],
    fps: float = 1.0,
    system_prompt: Optional[str] = None,
    default_question: str = "Find all significant events in the video and describe them.",
    video_prefix: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert from format.json to stream format with <stream> tokens.
    
    Input (format.json):
    {
        "video_path": "/path/to/video.mp4",
        "question": [{"content": "...", "time": "0"}],
        "response": [{"content": "...", "st_time": "0", "end_time": "5"}]
    }
    
    Output:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Question<stream>"},
            {"role": "assistant", "content": "</Standby>"},
            ...
        ],
        "videos": ["/path/to/video.mp4"]
    }
    """
    # Get video path
    video_path = data.get('video_path', '')
    if not video_path:
        videos = data.get('videos', [])
        if videos:
            video_path = videos[0] if isinstance(videos, list) else videos
    
    # Apply video prefix if provided
    if video_prefix and video_path:
        video_path = os.path.join(video_prefix, video_path)
    
    if not video_path or not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return None
    
    # Get frame count
    try:
        num_frames = get_video_frame_count(video_path, fps)
    except Exception as e:
        print(f"Warning: Cannot read video {video_path}: {e}")
        return None
    
    if num_frames <= 0:
        print(f"Warning: No frames in video: {video_path}")
        return None
    
    # Get questions and responses
    questions = data.get('question', [])
    responses = data.get('response', [])
    
    # Build frame states
    frame_states, question_map = build_frame_states(
        num_frames, questions, responses, fps
    )
    
    # Set default question at frame 0 if no question specified
    if 0 not in question_map:
        if questions and questions[0].get('content'):
            question_map[0] = questions[0]['content']
        else:
            question_map[0] = default_question
    
    # Build messages
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": system_prompt or DEFAULT_SYSTEM_PROMPT
    })
    
    # Frame-by-frame messages with <stream> token
    for i in range(num_frames):
        # User message with time format: <Xs-Ys>
        time_tag = f"<{i}s-{i+1}s>"
        if i in question_map:
            user_content = f"{question_map[i]}\n{time_tag}\n{STREAM_TOKEN}"
        else:
            user_content = f"{time_tag}\n{STREAM_TOKEN}"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Assistant message
        messages.append({
            "role": "assistant",
            "content": frame_states[i]
        })
    
    result = {
        "messages": messages,
        "videos": [video_path]
    }
    
    # Preserve metadata
    for key in ['video_name', 'task_type', 'source']:
        if key in data and data[key]:
            result[key] = data[key]
    
    return result


def convert_stream_to_image(
    data: Dict[str, Any],
    fps: float = 1.0,
    frame_output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert from stream format to image format (extract frames).
    
    Replaces <stream> with <image> and extracts video frames.
    """
    from swift.llm.dataset.preprocessor import StreamingVideoPreprocessor
    
    preprocessor = StreamingVideoPreprocessor(
        fps=fps,
        max_frames=max_frames,
        frame_output_dir=frame_output_dir,
        use_frame_cache=True,
    )
    
    return preprocessor.preprocess(data)


def _convert_to_stream_worker(
    args: Tuple[int, Dict[str, Any], float, Optional[str], Optional[str]]
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Worker function for parallel to-stream conversion."""
    idx, item, fps, system_prompt, video_prefix = args
    result = convert_format_to_stream(
        item,
        fps=fps,
        system_prompt=system_prompt,
        video_prefix=video_prefix,
    )
    return idx, result


def batch_convert_to_stream(
    input_path: str,
    output_path: str,
    fps: float = 1.0,
    system_prompt: Optional[str] = None,
    video_prefix: Optional[str] = None,
    num_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Batch convert from format.json to stream format with multi-threading.
    """
    from tqdm import tqdm
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle single object vs list
    is_single = isinstance(data, dict)
    data_list = [data] if is_single else data
    
    # Prepare worker arguments
    worker_args = [
        (idx, item, fps, system_prompt, video_prefix)
        for idx, item in enumerate(data_list)
    ]
    
    # Use thread pool for parallel processing
    results_dict = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_convert_to_stream_worker, args): args[0]
            for args in worker_args
        }
        
        with tqdm(total=len(data_list), desc="Converting to stream format") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                if result:
                    results_dict[idx] = result
                pbar.update(1)
    
    # Sort results by original index to maintain order
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    
    # Output
    output_data = results[0] if is_single and len(results) == 1 else results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nConverted {len(results)} samples to stream format")
    print(f"Output: {output_path}")
    
    # Statistics
    if results:
        total_rounds = sum(
            sum(1 for m in r['messages'] if m['role'] == 'user')
            for r in results
        )
        print(f"Total frames/rounds: {total_rounds}")
    
    return results


def _convert_to_image_worker(
    args: Tuple[int, Dict[str, Any], float, Optional[str], Optional[int]]
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Worker function for parallel to-image conversion."""
    idx, item, fps, frame_output_dir, max_frames = args
    result = convert_stream_to_image(
        item,
        fps=fps,
        frame_output_dir=frame_output_dir,
        max_frames=max_frames,
    )
    return idx, result


def batch_convert_to_image(
    input_path: str,
    output_path: str,
    fps: float = 1.0,
    frame_output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
    num_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Batch convert from stream format to image format with multi-threading.
    """
    from tqdm import tqdm
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle single object vs list
    is_single = isinstance(data, dict)
    data_list = [data] if is_single else data
    
    # Prepare worker arguments
    worker_args = [
        (idx, item, fps, frame_output_dir, max_frames)
        for idx, item in enumerate(data_list)
    ]
    
    # Use thread pool for parallel processing
    results_dict = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_convert_to_image_worker, args): args[0]
            for args in worker_args
        }
        
        with tqdm(total=len(data_list), desc="Converting to image format") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                if result:
                    results_dict[idx] = result
                pbar.update(1)
    
    # Sort results by original index to maintain order
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    
    # Output
    output_data = results[0] if is_single and len(results) == 1 else results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nConverted {len(results)} samples to image format")
    print(f"Output: {output_path}")
    
    # Statistics
    if results:
        total_frames = sum(len(r.get('images', [])) for r in results)
        print(f"Total frames extracted: {total_frames}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert streaming video data between formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Conversion mode')
    
    # to-stream: Convert format.json to stream format
    stream_parser = subparsers.add_parser(
        'to-stream',
        help='Convert from format.json to videos+<stream> multi-round format'
    )
    stream_parser.add_argument('--input', '-i', required=True, help='Input JSON file (format.json)')
    stream_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    stream_parser.add_argument('--fps', type=float, default=1.0, help='FPS for frame counting')
    stream_parser.add_argument('--system-prompt', help='Custom system prompt')
    stream_parser.add_argument('--video-prefix', help='Prefix path to prepend to video_path')
    stream_parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers')
    
    # to-image: Convert stream format to image format
    image_parser = subparsers.add_parser(
        'to-image',
        help='Convert from videos+<stream> to images+<image> format (extract frames)'
    )
    image_parser.add_argument('--input', '-i', required=True, help='Input JSON file (stream format)')
    image_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    image_parser.add_argument('--fps', type=float, default=1.0, help='FPS for frame extraction')
    image_parser.add_argument('--frame-dir', help='Directory to save extracted frames')
    image_parser.add_argument('--max-frames', type=int, help='Maximum frames per video')
    image_parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'to-stream':
        batch_convert_to_stream(
            input_path=args.input,
            output_path=args.output,
            fps=args.fps,
            system_prompt=args.system_prompt,
            video_prefix=args.video_prefix,
            num_workers=args.num_workers,
        )
    elif args.command == 'to-image':
        batch_convert_to_image(
            input_path=args.input,
            output_path=args.output,
            fps=args.fps,
            frame_output_dir=args.frame_dir,
            max_frames=args.max_frames,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
