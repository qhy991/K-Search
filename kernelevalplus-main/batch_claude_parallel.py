#!/usr/bin/env python3
"""
并行批量处理脚本 - 支持同时运行多个任务
使用 tmux 的多个窗口来实现并行
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import threading
import queue


class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


def log(msg: str, color: str = Colors.GREEN):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{color}[{timestamp}]{Colors.NC} {msg}")


def warn(msg: str):
    log(msg, Colors.YELLOW)


def error(msg: str):
    log(msg, Colors.RED)


def info(msg: str):
    log(msg, Colors.BLUE)


class TaskTracker:
    """任务进度跟踪器 - 支持断点续传（线程安全）"""

    def __init__(self, state_file: str, exp_dir: str = None):
        self.state_file = state_file
        self.exp_dir = exp_dir  # 参考实验目录
        self.lock = threading.Lock()
        self.state = self._load_state()

        # 如果提供了实验目录，扫描已有的实验结果
        if self.exp_dir and os.path.exists(self.exp_dir):
            self._scan_existing_experiments()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warn(f"无法加载状态文件: {e}")
        return {
            "completed": [],
            "failed": [],
            "running": [],
            "current_task": None,
            "started_at": None
        }

    def _scan_existing_experiments(self):
        """扫描已有实验目录，将已完成的实验添加到状态中"""
        if not self.exp_dir or not os.path.exists(self.exp_dir):
            return

        info(f"扫描已有实验目录: {self.exp_dir}")
        scanned_count = 0

        # 遍历实验目录下的所有子目录
        for root, dirs, files in os.walk(self.exp_dir):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # 检查目录中是否有内核文件（.cu文件）
                has_kernel = False
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path) and item.endswith('.cu'):
                        has_kernel = True
                        break
                    elif os.path.isdir(item_path):
                        # 检查子目录中是否有.cu文件
                        for sub_root, sub_dirs, sub_files in os.walk(item_path):
                            if any(f.endswith('.cu') for f in sub_files):
                                has_kernel = True
                                break
                    if has_kernel:
                        break

                if has_kernel:
                    # 使用目录路径作为标识（相对路径）
                    rel_path = os.path.relpath(dir_path, self.exp_dir)
                    if rel_path not in self.state["completed"]:
                        self.state["completed"].append(rel_path)
                        scanned_count += 1

        if scanned_count > 0:
            info(f"从已有实验目录中识别出 {scanned_count} 个已完成的实验")
            self._save_state_nolock()

    def _save_state(self):
        with self.lock:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)

    def mark_started(self, task_file: str):
        with self.lock:
            self.state["current_task"] = task_file
            if self.state["started_at"] is None:
                self.state["started_at"] = datetime.now().isoformat()
            if task_file not in self.state["running"]:
                self.state["running"].append(task_file)
            self._save_state_nolock()

    def mark_completed(self, task_file: str):
        with self.lock:
            if task_file not in self.state["completed"]:
                self.state["completed"].append(task_file)
            if task_file in self.state["running"]:
                self.state["running"].remove(task_file)
            self.state["current_task"] = None
            self._save_state_nolock()

    def mark_failed(self, task_file: str):
        with self.lock:
            if task_file not in self.state["failed"]:
                self.state["failed"].append(task_file)
            if task_file in self.state["running"]:
                self.state["running"].remove(task_file)
            self.state["current_task"] = None
            self._save_state_nolock()

    def _save_state_nolock(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_completed(self, task_file: str, exp_dir_mapping: Dict[str, str] = None) -> bool:
        with self.lock:
            # 检查状态文件中的记录
            if task_file in self.state["completed"]:
                return True

            # 如果提供了实验目录映射，检查实验目录中是否存在
            if self.exp_dir and exp_dir_mapping:
                task_name = Path(task_file).stem
                # 根据任务类型确定子目录
                for exp_subdir, task_patterns in exp_dir_mapping.items():
                    exp_path = os.path.join(self.exp_dir, exp_subdir, task_name)
                    if os.path.exists(exp_path):
                        # 检查目录中是否有内核文件
                        if self._has_kernel_files(exp_path):
                            info(f"  [跳过] {task_name} - 已在 {exp_path} 完成")
                            # 添加到已完成列表
                            if task_file not in self.state["completed"]:
                                self.state["completed"].append(task_file)
                            self._save_state_nolock()
                            return True

            return False

    def _has_kernel_files(self, directory: str) -> bool:
        """检查目录中是否有内核文件（.cu文件）"""
        if not os.path.exists(directory):
            return False
        for root, dirs, files in os.walk(directory):
            if any(f.endswith('.cu') for f in files):
                return True
        return False

    def is_failed(self, task_file: str) -> bool:
        with self.lock:
            return task_file in self.state["failed"]

    def get_summary(self) -> str:
        with self.lock:
            completed = len(self.state["completed"])
            failed = len(self.state["failed"])
            running = len(self.state["running"])
            total = completed + failed + running
            return f"运行中: {running}, 已完成: {completed}, 失败: {failed}, 总计: {total}"


class ParallelTaskProcessor:
    """并行任务处理器"""

    def __init__(self, args):
        self.task_dir = args.task_dir
        self.max_workers = args.workers
        self.delay_after_cmd = args.delay
        self.output_dir = args.output_dir  # 保存原始输出目录路径
        self.output_base = args.output_dir
        self.dry_run = args.dry_run
        self.skip_completed = args.skip_completed
        self.retry_failed = args.retry_failed
        self.exp_dir = getattr(args, 'exp_dir', None)  # 参考实验目录（必须在tracker初始化前设置）

        # 构建环境变量列表
        self.env_vars = self._build_env_vars(args)

        # 构建输出目录路径 - 统一格式: output/{model}/
        model_group = args.model_group or args.model
        self.model_name = model_group  # 保存模型名称

        # 从任务目录提取任务类型
        task_dir_name = Path(args.task_dir).name
        self.task_type = self._extract_task_type_from_dir(task_dir_name)  # 保存任务类型

        # 统一输出目录: output/{model}/
        # 具体任务目录: output/{model}/{op_type}/{task_name}/
        self.output_base = os.path.join(self.output_base, model_group)

        # 自动生成 tmux 会话名: claude-{model}-{task}
        self.base_session = self._generate_session_name(args)

        # 状态文件需要同时考虑 model 和 model_group，确保不同模型的实验状态分离
        # 即使 model_group 相同，如果 model 不同，也应该使用不同的状态文件
        model_for_state = args.model.lower().replace(".", "").replace("-", "")
        if len(model_for_state) > 10:
            model_for_state = model_for_state[:10]
        task_short = self._task_short_name(task_dir_name)
        state_suffix = f"{model_for_state}-{task_short}"
        self.state_file = os.path.join(self.output_base, f".batch_{state_suffix}.json")

        os.makedirs(self.output_base, exist_ok=True)
        self.tracker = TaskTracker(self.state_file, self.exp_dir)

        # 构建实验目录映射（任务类型 -> 实验子目录）
        self.exp_dir_mapping = self._build_exp_dir_mapping()

        info(f"TMUX 会话: {self.base_session}")
        info(f"输出目录: {self.output_base}")
        if self.exp_dir:
            info(f"参考实验目录: {self.exp_dir}")

    def _generate_session_name(self, args) -> str:
        """根据模型和任务类型生成 tmux 会话名"""
        if args.session != "claude-parallel":
            return args.session

        # 从任务目录提取任务类型
        task_dir = Path(args.task_dir).name
        task_short = self._task_short_name(task_dir)

        # 优先使用 model_group，否则使用 model
        model_name = args.model_group or args.model
        model_short = model_name.lower().replace(".", "").replace("-", "")
        # 限制长度
        if len(model_short) > 10:
            model_short = model_short[:10]

        return f"claude-{model_short}-{task_short}"

    def _task_short_name(self, task_dir: str) -> str:
        """任务目录简称"""
        task_map = {
            "flash_attention": "fa",
            "quant_gemm": "qg",
            "topk": "tk",
            "rms_norm": "rn",
            "rope": "rp",
            "moe": "moe",
        }
        for key, short in task_map.items():
            if key in task_dir.lower():
                return short
        # 默认取前2个字符
        return task_dir[:2].lower()

    def _build_env_vars(self, args) -> List[str]:
        """根据参数构建环境变量列表"""
        model_name = args.model_group or args.model
        return [
            f"export ANTHROPIC_BASE_URL={args.base_url}",
            f"export ANTHROPIC_AUTH_TOKEN={args.auth_token}",
            f"export ANTHROPIC_MODEL={args.model}",
            f"export ANTHROPIC_DEFAULT_OPUS_MODEL={args.model}",
            f"export ANTHROPIC_DEFAULT_SONNET_MODEL={args.model}",
            f"export ANTHROPIC_DEFAULT_HAIKU_MODEL={args.model}",
            f"export CLAUDE_CODE_SUBAGENT_MODEL={args.model}",
            f"export KERNEL_MODEL={model_name}",  # 传递模型名称给 skill
        ]

    def _build_exp_dir_mapping(self) -> Dict[str, str]:
        """构建任务类型到实验子目录的映射"""
        # 根据任务类型返回对应的实验子目录
        mapping = {
            "flash_attention": "flash_attention",
            "quant_gemm": "quant_gemm",
            "topk": "glm-5-topk-20260309",  # 根据实际情况调整
            "rms_norm": "glm-5-topk-20260309",  # 根据实际情况调整
        }
        # 只返回当前任务类型的映射
        return {self.task_type: mapping.get(self.task_type, self.task_type)}

    def find_tasks(self) -> List[Path]:
        task_path = Path(self.task_dir)
        if not task_path.exists():
            error(f"任务路径不存在: {self.task_dir}")
            sys.exit(1)

        # 支持单个JSON文件或目录
        if task_path.is_file():
            if task_path.suffix != ".json":
                error(f"单个文件必须是JSON格式: {self.task_dir}")
                sys.exit(1)
            tasks = [task_path]
        else:
            # 目录模式：递归查找所有JSON文件
            tasks = list(task_path.rglob("*.json"))

        tasks.sort()

        filtered_tasks = []
        for task in tasks:
            if self.skip_completed and self.tracker.is_completed(str(task), self.exp_dir_mapping):
                continue
            if not self.retry_failed and self.tracker.is_failed(str(task)):
                continue
            filtered_tasks.append(task)

        return filtered_tasks

    def ensure_base_session(self) -> bool:
        """确保基础 tmux 会话存在"""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.base_session],
                capture_output=True
            )
            if result.returncode != 0:
                info(f"创建新的 tmux 会话: {self.base_session}")
                subprocess.run(
                    ["tmux", "new-session", "-d", "-s", self.base_session],
                    check=True
                )
                time.sleep(2)
            return True
        except subprocess.CalledProcessError as e:
            error(f"无法创建 tmux 会话: {e}")
            return False

    def send_cmd_to_window(self, window_name: str, cmd: str, delay: int = None):
        """发送命令到指定窗口"""
        if delay is None:
            delay = self.delay_after_cmd

        target = f"{self.base_session}:{window_name}"

        if self.dry_run:
            print(f"[DRY RUN][{window_name}] {cmd[:80]}...")
            return

        try:
            # 分两步：先发送命令，再发送回车
            subprocess.run(
                ["tmux", "send-keys", "-t", target, cmd],
                check=True
            )
            time.sleep(0.2)
            subprocess.run(
                ["tmux", "send-keys", "-t", target, "Enter"],
                check=True
            )
            time.sleep(delay)
        except subprocess.CalledProcessError as e:
            error(f"[{window_name}] 发送命令失败: {e}")

    def setup_window(self, window_name: str):
        """设置窗口环境"""
        info(f"[{window_name}] 设置环境...")
        self.send_cmd_to_window(window_name, "cd /home/qinhaiyan/kernelevalplus", 1)
        for env_var in self.env_vars:
            self.send_cmd_to_window(window_name, env_var, 1)

    def create_window(self, window_name: str) -> bool:
        """创建新的 tmux 窗口（如果已存在则先清理）"""
        if self.dry_run:
            info(f"[DRY RUN] 创建窗口: {window_name}")
            return True

        try:
            target = f"{self.base_session}:{window_name}"
            # 检查窗口是否已存在
            result = subprocess.run(
                ["tmux", "list-windows", "-t", self.base_session],
                capture_output=True, text=True
            )
            if window_name in result.stdout:
                info(f"[{window_name}] 窗口已存在，清理后重新创建")
                # 先 kill 旧窗口
                subprocess.run(
                    ["tmux", "kill-window", "-t", target],
                    check=False
                )
                time.sleep(0.5)

            subprocess.run(
                ["tmux", "new-window", "-t", self.base_session, "-n", window_name],
                check=True
            )
            time.sleep(1)
            return True
        except subprocess.CalledProcessError as e:
            error(f"创建窗口 {window_name} 失败: {e}")
            return False

    def process_single_task(self, task_file: Path, worker_id: int) -> bool:
        """处理单个任务（在独立窗口中）"""
        task_name = task_file.stem
        task_path = str(task_file)
        window_name = f"worker-{worker_id}"

        # 统一输出目录: output/{model}/{op_type}/{task_name}/
        output_dir = os.path.join(self.output_base, self.task_type, task_name)

        os.makedirs(output_dir, exist_ok=True)

        info(f"[{window_name}] 开始处理: {task_name}")
        info(f"[{window_name}] 输出目录: {output_dir}")
        self.tracker.mark_started(task_path)

        try:
            # 创建/获取窗口（会先清理已存在的窗口）
            if not self.create_window(window_name):
                return False

            # 设置环境
            self.setup_window(window_name)

            # Step 1: 初始求解 (40分钟)
            info(f"[{window_name}] Step 1: 初始求解")
            self.send_cmd_to_window(
                window_name,
                f"echo '=== [{task_name}] Step 1: Initial solution ==='", 2
            )
            self.send_cmd_to_window(
                window_name,
                f"claude --dangerously-skip-permissions \"use cuda-kernel-development skill to solve '{task_path}'\"",
                delay=2400  # 40分钟
            )

            # Step 2: 性能优化 (40分钟)
            info(f"[{window_name}] Step 2: 性能优化")
            self.send_cmd_to_window(
                window_name,
                f"echo '=== [{task_name}] Step 2: Performance improvement ==='", 2
            )
            self.send_cmd_to_window(
                window_name,
                "can you improve the performance based on the skill",
                delay=2400  # 40分钟
            )

            # Step 3: 保存文档、整理代码并移动结果
            info(f"[{window_name}] Step 3: 保存文档、整理代码并移动结果")
            self.send_cmd_to_window(
                window_name,
                f"echo '=== [{task_name}] Step 3: Save, organize and move ==='", 2
            )

            # 生成目标移动路径 (统一使用下划线)
            date_str = datetime.now().strftime("%Y%m%d")
            exp_target_dir = f"/home/qinhaiyan/KERNELEVAL-exp/{self.model_name}_{self.task_type}_{date_str}"

            # 统一提示：保存 + 移动
            self.send_cmd_to_window(
                window_name,
                f"请完成以下任务：\n"
                f"1. 总结优化历程，保存到文档 {output_dir}/summary.md\n"
                f"2. 测试时请将 attempt 保存到 {output_dir}/attempts/v{{N}}/ 目录下\n"
                f"3. 确保所有文件保存在 {output_dir} 文件夹下\n"
                f"4. 把效果最好的版本挑出来\n"
                f"5. 完成后，将整个输出目录移动到 {exp_target_dir}/\n"
                f"   即: mv {self.output_base} {exp_target_dir}/",
                delay=600
            )

            self.tracker.mark_completed(task_path)
            info(f"[{window_name}] 任务 {task_name} 完成")
            return True

        except Exception as e:
            error(f"[{window_name}] 任务 {task_name} 失败: {e}")
            self.tracker.mark_failed(task_path)
            return False

    def _extract_task_type_from_dir(self, task_dir: str) -> str:
        """从任务目录名称提取任务类型"""
        task_dir_lower = task_dir.lower()
        if "flash_attention" in task_dir_lower or "flash-attention" in task_dir_lower:
            return "flash_attention"
        elif "quant_gemm" in task_dir_lower or "quant-gemm" in task_dir_lower:
            return "quant_gemm"
        elif "topk" in task_dir_lower:
            return "topk"
        elif "rms_norm" in task_dir_lower or "rms-norm" in task_dir_lower:
            return "rms_norm"
        elif "rope" in task_dir_lower:
            return "rope"
        elif "moe" in task_dir_lower:
            return "moe"
        return "other"

    def _extract_task_type(self, task_path: str) -> str:
        """从任务路径提取任务类型"""
        path_parts = Path(task_path).parts
        for part in reversed(path_parts):
            if part in ["flash_attention", "quant_gemm", "topk", "rms_norm", "rope", "moe"]:
                return part
        return "other"

    def run(self):
        """运行并行批量处理"""
        tasks = self.find_tasks()

        if not tasks:
            warn("没有找到需要处理的任务")
            print(self.tracker.get_summary())
            return

        info(f"找到 {len(tasks)} 个待处理任务")
        info(f"并行工作线程数: {self.max_workers}")

        print(f"\n即将处理以下任务:")
        for i, task in enumerate(tasks[:10], 1):
            print(f"  {i}. {task}")
        if len(tasks) > 10:
            print(f"  ... 还有 {len(tasks) - 10} 个任务")

        response = input("\n是否开始处理? (y/N): ")
        if response.lower() != 'y':
            info("取消执行")
            return

        # 确保 tmux 会话存在
        if not self.ensure_base_session():
            error("无法创建 tmux 会话")
            sys.exit(1)

        start_time = time.time()

        # 使用线程池并行处理
        from concurrent.futures import ThreadPoolExecutor, as_completed

        completed_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="worker") as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task in enumerate(tasks):
                worker_id = i % self.max_workers
                future = executor.submit(self.process_single_task, task, worker_id)
                future_to_task[future] = task

            # 等待任务完成
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    if future.result():
                        completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    error(f"任务异常 {task}: {e}")
                    failed_count += 1

        # 计算总耗时
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        info("=========================================")
        info("并行批量处理完成!")
        info(f"成功: {completed_count}, 失败: {failed_count}")
        info(f"总耗时: {hours}h {minutes}m {seconds}s")
        info(f"状态文件: {self.state_file}")
        info(f"输出目录: {self.output_base}")
        info("=========================================")

        print(f"\n查看 tmux 会话: tmux attach -t {self.base_session}")
        print("切换窗口: Ctrl+B 然后按数字键 0-9")


def main():
    parser = argparse.ArgumentParser(
        description="Claude CUDA Kernel Development 并行批量处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 并行处理 10 个任务
  %(prog)s <task_dir> -w 10

  # 使用 minimax-m2.5 模型
  %(prog)s <task_dir> --model minimax-m2.5

  # 使用自定义延迟时间（秒）
  %(prog)s <task_dir> -w 5 -d 3600

  # 断点续传
  %(prog)s <task_dir> -w 10 --skip-completed

  # 使用自定义 API 配置
  %(prog)s <task_dir> --base-url https://open.bigmodel.cn/api/anthropic --auth-token xxx --model GLM-4.7

  # 按模型分组输出 (输出到 output/minimax-m2.5/flash_attention)
  %(prog)s <task_dir> --model minimax-m2.5 --model-group minimax-m2.5

  # 跳过已有实验目录中已完成的实验
  %(prog)s <task_dir> --model GLM-5 --skip-completed --exp-dir /home/qinhaiyan/KERNELEVAL-exp/GLM_5

  # 查看各个窗口
  tmux attach -t claude-parallel
  # 切换窗口: Ctrl+B 然后按数字键
        """
    )

    parser.add_argument("task_dir", help="任务目录路径")
    parser.add_argument("-s", "--session", default="claude-parallel",
                        help="tmux 会话名 (默认: claude-parallel)")
    parser.add_argument("-w", "--workers", type=int, default=10,
                        help="并行工作线程数 (默认: 10)")
    parser.add_argument("-d", "--delay", type=int, default=3600,
                        help="每个命令后等待秒数 (默认: 3600 = 1小时)")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="输出目录 (默认: output)")

    # API 配置参数
    parser.add_argument("--base-url", default="https://cloud.infini-ai.com/maas",
                        help="Anthropic API 基础 URL")
    parser.add_argument("--auth-token", default="sk-3na7x5s24w3fqnvt",
                        help="Anthropic API 认证令牌")
    parser.add_argument("--model", default="glm-5",
                        help="使用的模型名称 (如 glm-5, minimax-m2.5)")
    parser.add_argument("--model-group", default=None,
                        help="模型分组名称，用于区分不同模型的实验 (如 glm-5, minimax-m2.5)")

    parser.add_argument("--skip-completed", action="store_true",
                        help="跳过已完成的任务")
    parser.add_argument("--retry-failed", action="store_true",
                        help="重试失败的任务")
    parser.add_argument("--dry-run", action="store_true",
                        help="模拟运行")
    parser.add_argument("--exp-dir", default=None,
                        help="参考实验目录（用于检查已完成的实验）")

    args = parser.parse_args()

    processor = ParallelTaskProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()
