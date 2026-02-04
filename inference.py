import os
from typing import List
import cv2
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MIN_PIXELS'] = '3136'
os.environ['MAX_PIXELS'] = '100352'


SYSTEM = """
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


class VideoFrameExtractor:
    """Extract frames from video file at specified fps"""
    
    def __init__(self, video_path: str, target_fps: float = 1.0):
        """
        Args:
            video_path: Path to the video file
            target_fps: Target frame rate, default 1fps (1 frame per second)
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0
        
        # Calculate frame extraction interval
        self.frame_interval = int(self.original_fps / self.target_fps)
        self.num_extracted_frames = int(self.duration * self.target_fps)
        
        print(f"Video info: {video_path}")
        print(f"  Original FPS: {self.original_fps:.2f}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Duration: {self.duration:.2f}s")
        print(f"  Target FPS: {self.target_fps}")
        print(f"  Frames to extract: {self.num_extracted_frames}")
    
    def get_frame_at_time(self, time_sec: float) -> Image.Image:
        """Get frame at specified time point"""
        frame_idx = int(time_sec * self.original_fps)
        frame_idx = min(frame_idx, self.total_frames - 1)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"Cannot read frame at time {time_sec}s (frame {frame_idx})")
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def get_frame_at_round(self, round_num: int) -> Image.Image:
        """Get frame at specified round (each round corresponds to 1 second)"""
        time_sec = round_num  # Each round corresponds to 1 second
        return self.get_frame_at_time(time_sec)
    
    def get_total_rounds(self) -> int:
        """Get total number of rounds (based on video duration and target fps)"""
        return self.num_extracted_frames
    
    def close(self):
        """Release video resources"""
        if self.cap is not None:
            self.cap.release()
    
    def __del__(self):
        self.close()


def infer_single(engine: 'InferEngine', infer_requests: 'InferRequest'):
    request_config = RequestConfig(max_tokens=512, temperature=0.0)
    metric = InferStats()
    resp_list = engine.infer([infer_requests], request_config, metrics=[metric])
    response = resp_list[0].choices[0].message.content
    return response


def get_data_stream_video(
    video_extractor: VideoFrameExtractor,
    round_num: int,
    system: str = None,
    question: str = None,
    question_time: int = 0,
    data: dict = None,
    answer: str = None
) -> dict:
    """
    Build multi-turn conversation data by extracting frames directly from video
    
    Args:
        video_extractor: Video frame extractor
        round_num: Current round number
        system: System prompt
        question: Question to ask
        question_time: Time point when question appears
        data: Existing conversation data
        answer: Answer from previous round
    """
    # Get frame for current round
    frame = video_extractor.get_frame_at_round(round_num)
    
    if data is None:
        data = {}
        if round_num != 0:
            raise ValueError("round_num must be 0 when data is None")
        messages = [
            {'role': 'system', 'content': system},
        ]
        
        if round_num == question_time:
            messages.append({'role': 'user', 'content': f'{question}\n<{round_num}s-{int(round_num)+1}s>\n<image>'})
        else:
            messages.append({'role': 'user', 'content': f"<{round_num}s-{int(round_num)+1}s>\n<image>"})
        
        data['images'] = [frame]  # Directly use PIL.Image object
        data['messages'] = messages
        return data
    else:
        messages = data['messages']
        messages.append({'role': 'assistant', 'content': answer})
        if round_num == question_time:
            messages.append({'role': 'user', 'content': f'{question}\n<{round_num}s-{int(round_num)+1}s>\n<image>'})
        else:
            messages.append({'role': 'user', 'content': f"<{round_num}s-{int(round_num)+1}s>\n<image>"})
        
        data['images'].append(frame)
        data['messages'] = messages
        return data


def get_data_stream_video_window(
    video_extractor: VideoFrameExtractor,
    round_num: int,
    system: str = None,
    question: str = None,
    question_time: int = 0,
    data: dict = None,
    answer: str = None,
    max_rounds: int = 120,
    global_question: bool = False
) -> dict:
    """
    Build multi-turn conversation data by extracting frames directly from video, with sliding window
    
    Args:
        video_extractor: Video frame extractor
        round_num: Current round number
        system: System prompt
        question: Question to ask
        question_time: Time point when question appears
        data: Existing conversation data
        answer: Answer from previous round
        max_rounds: Maximum number of rounds to keep
        global_question: If True, the first user message after truncation always includes the question
    
    Returns:
        dict: Conversation data containing 'images' and 'messages'
    """
    # Get frame for current round
    frame = video_extractor.get_frame_at_round(round_num)
    
    def make_user_content(r: int, include_question: bool = False) -> str:
        time_tag = f"<{r}s-{r + 1}s>\n<image>"
        if include_question and question:
            return f"{question}\n{time_tag}"
        return time_tag
    
    if data is None:
        # Initialize data
        if round_num != 0:
            raise ValueError("round_num must be 0 when data is None")
        
        data = {}
        messages = []

        if system:
            messages.append({'role': 'system', 'content': system})
        
        # Add first user message
        include_q = global_question or (round_num == question_time)
        messages.append({
            'role': 'user', 
            'content': make_user_content(round_num, include_q)
        })
        
        data['images'] = [frame]
        data['messages'] = messages
        return data
    
    else:
        messages = data['messages']
        
        # Add answer from previous round
        if answer is not None:
            messages.append({'role': 'assistant', 'content': answer})
        
        # Add current round's user message
        # When adding normally, only include question when question_time matches (non-truncation scenario)
        include_q = (round_num == question_time)
        messages.append({
            'role': 'user', 
            'content': make_user_content(round_num, include_q)
        })
        
        # Add current frame
        data['images'].append(frame)
        
        # If exceeding max_rounds, perform sliding window truncation
        if len(data['images']) > max_rounds:
            rounds_to_remove = len(data['images']) - max_rounds
            
            start_round = round_num - max_rounds + 1
            
            new_messages = messages[:1]
            messages_to_skip = rounds_to_remove * 2  # Each round has user and assistant messages
            new_messages.extend(messages[1 + messages_to_skip:])
            
            include_q_start = global_question or (question_time == start_round)
            new_messages[1] = {
                'role': 'user',
                'content': make_user_content(start_round, include_q_start)
            }
            
            new_images = data['images'][rounds_to_remove:]
            
            data['messages'] = new_messages
            data['images'] = new_images
        
        return data


if __name__ == '__main__':
    from swift.llm import InferRequest, PtEngine, RequestConfig
    from swift.plugin import InferStats
    import json

    infer_backend = 'vllm'
    model = 'MODEL_PATH'
    
    if infer_backend == 'pt':
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model, max_model_len=32768, limit_mm_per_prompt={'image': 500}, tensor_parallel_size=1, enable_prefix_caching=True)

    video_path = './demo/cook.mp4'
    
    target_fps = 1.0
    video_extractor = VideoFrameExtractor(video_path, target_fps=target_fps)
    
    # question = 'What is being added to the bowl?'
    question = 'Detect and summarize each event sequence in the video.'
    global_question = True
    system = SYSTEM
    question_time = 0
    max_rounds = 300
    
    # Get total number of rounds
    round_num = video_extractor.get_total_rounds()
    print(f"Total rounds: {round_num}")
    
    output = {}
    data = None
    
    for i in range(round_num):
        if i == 0:
            data = get_data_stream_video_window(
                video_extractor=video_extractor,
                round_num=i,
                system=system,
                question=question,
                question_time=question_time,
                max_rounds=max_rounds,
                global_question=global_question
            )
        else:
            data = get_data_stream_video_window(
                video_extractor=video_extractor,
                round_num=i,
                system=system,
                question=question,
                question_time=question_time,
                data=data,
                answer=answer,
                max_rounds=max_rounds,
                global_question=global_question
            )
        
        if i < question_time:
            answer = "</Silence>"
        else:
            infer_request = InferRequest(**data)
            answer = infer_single(engine, infer_request)
        
        output[f"Round {i}"] = answer
        print("=====Round", i, "=====")
        print(f"Answer: {answer}")
    
    # Close video extractor
    video_extractor.close()
    
    with open('./test_sample_video.jsonl', 'a') as f:
        result = {
            "video_path": video_path,
            "target_fps": target_fps,
            "output": output
        }
        f.write(json.dumps(result) + '\n')
