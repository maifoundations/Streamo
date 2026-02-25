import os
import argparse
from typing import List, Literal, Dict, Any, Set, Union
import random
import glob
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import re
import json

os.environ['MIN_PIXELS'] = '3136'
os.environ['MAX_PIXELS'] = '100352'
os.environ['INPUT_SIZE'] = '448'
os.environ['MAX_NUM'] = '1'

VIDEO_PREFIX = ""

STOP_ON_RESPONSE_TASKS = {"EPM", "HLD", "ASI", "STU", "OJR", "ATR", "FPD", "OCR", "ACR", "CRR", "SSR"}

def _normalize_task_filter(task_filter: Union[None, Set[str], List[str], str]) -> Set[str]:
    if task_filter is None:
        return set()
    if isinstance(task_filter, str):
        task_filter = {t.strip() for t in task_filter.split(",") if t.strip()}
    elif isinstance(task_filter, (list, set)):
        task_filter = {str(t).strip() for t in task_filter if str(t).strip()}
    else:
        raise ValueError(
            "task_filter must be None, str, list or set, got: %s" % type(task_filter)
        )
    return task_filter

def load_dataset_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")
    
    processed_samples = []
    for sample in data:
        if sample.get("task") == "SSR" and "all_steps" in sample:
            all_steps = sample["all_steps"]
            start_times = sample.get("start_time", [])
            
            if isinstance(all_steps, str):
                steps = [step.strip() for step in all_steps.split('\n') if step.strip()]
            elif isinstance(all_steps, list):
                steps = all_steps
            else:
                processed_samples.append(sample)
                continue
            
            if isinstance(start_times, str):
                start_times = [int(t.strip()) for t in start_times.split(',') if t.strip()]
            elif not isinstance(start_times, list):
                start_times = []
            
            for i, step in enumerate(steps):
                new_sample = sample.copy()
                new_sample["all_steps"] = step
                new_sample["step_index"] = i
                new_sample["step_start_time"] = start_times[i] if i < len(start_times) else 0
                new_sample["original_id"] = sample["id"]
                new_sample["id"] = f"{sample['id']}_step_{i}"
                processed_samples.append(new_sample)
        else:
            processed_samples.append(sample)
    
    print(f"After processing SSR samples: {len(processed_samples)} samples")
    return processed_samples

def load_processed_ids_from_jsonl(jsonl_path: str) -> set:
    processed = set()
    if not jsonl_path or not os.path.exists(jsonl_path):
        return processed
    cnt = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("id") or obj.get("sample_id")
                if sid:
                    processed.add(sid)
                    cnt += 1
            except Exception as e:
                print(f"Warning: skip bad line in {jsonl_path}: {e}")
    print(f"Loaded {cnt} lines, found {len(processed)} processed ids from {jsonl_path}")
    return processed

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

class VideoProcessor:
    def __init__(self, sample: dict, system: str):
        self.sample = sample
        self.video_path = os.path.join(VIDEO_PREFIX, sample["video"])
        self.system = system
        self.current_round = 0
        self.data = None
        self.output = {}
        self.completed = False
        
        image_files = os.listdir(self.video_path)
        self.total_rounds = len(image_files) // 2
        
        self.question, self.question_time = self._build_question()
        
    def get_next_request(self, last_answer=None) -> tuple:
        if self.current_round == 0:
            self.data = get_data_stream_window(
                video=os.path.join(VIDEO_PREFIX, self.sample["video"]),
                round_num=self.current_round,
                system=self.system,
                question=self.question,
                question_time=self.question_time
            )
        else:
            self.data = get_data_stream_window(
                video=os.path.join(VIDEO_PREFIX, self.sample["video"]),
                round_num=self.current_round,
                system=self.system,
                question=self.question,
                data=self.data,
                answer=last_answer,
                question_time=self.question_time
            )
        
        should_process = self.current_round >= self.question_time
        
        if should_process:
            return True, InferRequest(**self.data)
        else:
            return False, None

    def _build_question(self) -> tuple:
        task = self.sample["task"]
        if task in {"EPM", "HLD", "STU", "OJR", "ATR", "FPD", "OCR", "ACR"}:
            tmpl = "{question}\n{options}\nPlease provide your answer by stating the letter followed by the full option."
            question_time = self.sample["realtime"]
            options = self.sample["options"]
            opts = "\n".join([f"{chr(65 + i)}. {o}" for i, o in enumerate(options)])
            return tmpl.format(question=self.sample["question"], options=opts), int(question_time)
        elif task in {"ASI"}:
            tmpl = "{question}\n{options}\nPlease provide your answer by stating the letter followed by the full option."
            question_time = random.randint(0, int(self.sample["realtime"]))
            options = self.sample["options"]
            opts = "\n".join([f"{chr(65 + i)}. {o}" for i, o in enumerate(options)])
            return tmpl.format(question=self.sample["question"], options=opts), int(question_time)
        elif task == "CRR":
            return self.sample["question"], self.sample["ask_time"]
        elif task == "SSR":
            question = self.sample["all_steps"]
            tmpl = "Watch the following video and temporally localize the event. Respond once it has finished and summarize its time period. The given event is:\n'{question}'"
            question_time = self.sample.get("step_start_time", 0)
            return tmpl.format(question=question), int(question_time)
        elif task == "REC":
            tmpl = "How many times did activity '{activity}' occur?\n{options}. Please respond with only the letter of the correct answer. Update your answer if it becomes different at a later time."
            
            options = '\n'.join(f"{chr(65 + i)}. {i+1}" for i in range(10))

            question_time = max(self.sample['start_times'][0] - 3, 0)
            return tmpl.format(activity=self.sample["activity"], options=options), int(question_time)
        else:
            return "", 0

    def update(self, answer: str):
        self.output[f"Round {self.current_round}"] = answer

        if self.sample.get("task") in STOP_ON_RESPONSE_TASKS and answer and "</Response>" in answer:
            self.current_round += 1
            self.completed = True
            return

        self.current_round += 1
        if self.current_round >= self.total_rounds:
            self.completed = True

    def is_completed(self) -> bool:
        return self.completed


def batch_infer(engine: 'InferEngine', infer_requests: List['InferRequest'], batch_size: int = 16):
    if not infer_requests:
        return []
        
    request_config = RequestConfig(max_tokens=512, temperature=0.1)
    
    all_responses = []
    for i in range(0, len(infer_requests), batch_size):
        batch_requests = infer_requests[i:i+batch_size]
        metric = InferStats()
        resp_list = engine.infer(batch_requests, request_config, metrics=[metric])
        
        batch_responses = [resp.choices[0].message.content for resp in resp_list]
        all_responses.extend(batch_responses)
        
        print(f'Batch {i//batch_size + 1}: processed {len(batch_requests)} requests')
        print(f'Metric: {metric.compute()}')
    
    return all_responses


def process_videos_batch(
    engine: "InferEngine",
    samples: List[dict],
    system: str,
    max_concurrent: int = 32,
    batch_size: int = 16,
    output_file: str = None,
    *,
    task_filter: Union[None, Set[str], List[str], str] = None,
):
    processed_ids = load_processed_ids_from_jsonl(output_file) if output_file else set()
    if processed_ids:
        orig = len(samples)
        samples = [s for s in samples if s.get("id") not in processed_ids]
        print(f"Startup filter: {orig - len(samples)} samples skipped (already processed), {len(samples)} remaining")

    task_set = _normalize_task_filter(task_filter)
    if task_set:
        orig = len(samples)
        samples = [s for s in samples if s.get("task") in task_set]
        print(f"Task filter: keep {len(samples)}/{orig} samples belonging to tasks {sorted(task_set)}")
    else:
        print("No task filter applied – all tasks will be processed.")

    if not samples:
        print("No samples to process after filtering. Exit.")
        return

    processors = {s["id"]: VideoProcessor(s, system) for s in samples}
    written_ids = set()
    eval_file = open(output_file, "a") if output_file else None

    def _advance_to_infer_or_complete(p: VideoProcessor, last_answer: str):
        skipped = 0
        la = last_answer
        while True:
            if p.is_completed():
                return None, skipped, la
            should_process, req = p.get_next_request(la)
            if should_process:
                return req, skipped, la
            p.update("</Silence>")
            skipped += 1
            la = "</Silence>"

    try:
        while True:
            active = {k: v for k, v in processors.items() if not v.is_completed()}
            if not active:
                print("All samples processed!")
                break
            print(f"\n--- Processing round ---\nActive: {len(active)}")

            batch_req, batch_keys = [], []
            for k, p in list(active.items())[:max_concurrent]:
                last_answer = p.output.get(f"Round {p.current_round-1}") if p.current_round else None
                req, skipped_cnt, _ = _advance_to_infer_or_complete(p, last_answer)

                if skipped_cnt:
                    print(
                        f"Sample-{k}: fast-forwarded {skipped_cnt} round(s) before question_time; "
                        f"now at round {p.current_round}/{p.total_rounds}"
                    )

                if req is not None:
                    batch_req.append(req)
                    batch_keys.append(k)

            if batch_req:
                responses = batch_infer(engine, batch_req, batch_size)
                for k, resp in zip(batch_keys, responses):
                    p = processors[k]
                    p.update(resp)
                    print(
                        f"Sample-{k}: id {p.sample['id']} round {p.current_round}/{p.total_rounds}, "
                        f"resp: {resp[:50]}..."
                    )

            for k, p in processors.items():
                if "gt" not in p.sample:
                    p.sample["gt"] = p.sample["test_info"]
                if p.is_completed() and eval_file and k not in written_ids:
                    output_data = {
                        "id": p.sample["id"],
                        "video": p.sample["video"],
                        "gt": p.sample["gt"],
                        "output": p.output,
                        "task": p.sample.get("task")
                    }
                    if p.sample.get("task") == "SSR":
                        output_data["step_index"] = p.sample.get("step_index", -1)
                        output_data["original_id"] = p.sample.get("original_id", "")
                        output_data["step_question"] = p.sample.get("all_steps", "")
                        output_data["step_start_time"] = p.sample.get("step_start_time", 0)
                    eval_file.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    eval_file.flush()
                    written_ids.add(k)

            completed = sum(1 for p in processors.values() if p.is_completed())
            print(f"Progress: {completed}/{len(processors)} completed")
    finally:
        if eval_file:
            eval_file.close()



def get_data_stream_window(video, round_num, system=None, question=None, data=None, answer=None, max_rounds=120, question_time=0):
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
        
        round_images = [
            f"{video}/frame_{round_num*2:06d}.png",
        ]
        data['images'] = round_images
        data['messages'] = messages
        return data
    else:
        messages = data['messages']
        messages.append({'role': 'assistant', 'content': answer})
        if round_num == question_time:
            messages.append({'role': 'user', 'content': f'{question}\n<{round_num}s-{int(round_num)+1}s>\n<image>'})
        else:
            messages.append({'role': 'user', 'content': f"<{round_num}s-{int(round_num)+1}s>\n<image>"})
        
        round_images = [
            f"{video}/frame_{round_num*2:06d}.png",
        ]
        data['images'].extend(round_images)
        
        if round_num > max_rounds:

            rounds_to_remove = 1
            
            new_messages = messages[:1]
            messages_to_skip = rounds_to_remove * 2
            
            new_messages.extend(messages[1 + messages_to_skip:])
            
            new_messages[1] = {'role': 'user', 'content': f'{question}\n<{round_num-max_rounds}s-{int(round_num)-max_rounds+1}s>\n<image>'}

            images_to_remove = rounds_to_remove
            new_images = data['images'][images_to_remove:]
            
            data['messages'] = new_messages
            data['images'] = new_images
        
        return data


def is_video_directory(directory_path):
    if not os.path.isdir(directory_path):
        return False
    
    frame_patterns = [
        r'frame_\d{6}\.(png|jpg|jpeg)',
        r'frame_\d{5}\.(png|jpg|jpeg)',
        r'frame_\d{4}\.(png|jpg|jpeg)',
        r'\d{6}\.(png|jpg|jpeg)',
        r'\d{5}\.(png|jpg|jpeg)',
    ]
    
    try:
        files = os.listdir(directory_path)
        frame_files = []
        
        for file in files:
            for pattern in frame_patterns:
                if re.match(pattern, file, re.IGNORECASE):
                    frame_files.append(file)
                    break
        
        return len(frame_files) >= 2
    
    except (PermissionError, OSError):
        return False


def find_video_directories(root_path, max_depth=10):
    video_dirs = []
    
    def _recursive_search(current_path, current_depth):
        if current_depth > max_depth:
            return
        
        try:
            if not os.path.isdir(current_path):
                return
            
            if is_video_directory(current_path):
                video_dirs.append(current_path)
                print(f"Found video directory: {current_path}")
                return
            
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    _recursive_search(item_path, current_depth + 1)
                    
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot access {current_path}: {e}")
    
    print(f"Searching for video directories in: {root_path}")
    _recursive_search(root_path, 0)
    print(f"Found {len(video_dirs)} video directories total")
    
    return video_dirs


if __name__ == "__main__":
    from swift.llm import (
        InferEngine,
        InferRequest,
        PtEngine,
        RequestConfig,
        VllmEngine,
    )
    from swift.plugin import InferStats

    parser = argparse.ArgumentParser(description="OVOBench video stream inference batch processing")
    parser.add_argument(
        "--tasks",
        type=str,
        default=["EPM", "HLD", "ASI", "STU", "OJR", "ATR", "FPD", "OCR", "SSR", "ACR", "CRR", "REC"],
        help="Comma-separated task list to process, e.g.: EPM,SSR. If not specified, all tasks will be processed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./test.jsonl",
        help="Output file path (jsonl)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=64,
        help="Maximum number of samples to process concurrently per round",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model inference",
    )
    args = parser.parse_args()

    infer_backend = "vllm"
    model = args.model
    if infer_backend == "pt":
        engine = PtEngine(model, max_batch_size=4, max_len=65536)
    elif infer_backend == "lmdeploy":
        engine = LmdeployEngine(model, vision_batch_size=64, tp=4)
    else:
        engine = VllmEngine(
            model,
            max_model_len=65536,
            limit_mm_per_prompt={"image": 500},
            tensor_parallel_size=4,
            enable_prefix_caching=True,
        )

    dataset_json = ""
    samples = load_dataset_json(dataset_json)

    system = SYSTEM

    process_videos_batch(
        engine=engine,
        samples=samples,
        system=system,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        output_file=args.output,
        task_filter=args.tasks,
    )