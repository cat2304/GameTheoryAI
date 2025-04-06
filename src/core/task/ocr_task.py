"""OCR任务管理模块: {
    "描述": "提供OCR任务的管理功能，包括GUI控制界面和后台任务执行。",
    "子模块": {
        "ocr_task": "OCR识别任务，包含图像预处理、文字识别、结果处理"
    },
    "主要功能": {
        "任务调度": ["任务创建", "任务优先级", "任务队列", "任务取消"],
        "任务执行": ["并行执行", "错误处理", "结果收集", "状态监控"]
    },
    "使用示例": "from src.core.task.ocr_task import OCRTask\n\n# 创建OCR任务\ntask = OCRTask(image_path='screenshot.png')\n\n# 执行任务\nresult = task.execute()\n\n# 获取结果\ntext = result.get_text()",
    "注意事项": ["任务执行需要合理配置资源", "注意任务优先级设置", "及时处理任务异常"]
}"""

# 版本信息
__version__ = '0.1.0'

import os
import sys
import time
import yaml
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import psutil
from pathlib import Path
from threading import Thread, Event
from queue import Queue

# 公共接口列表
__all__ = [
    'OCRTask'
]

class OCRTask:
    """OCR任务类，用于管理OCR识别任务"""
    
    def __init__(self, image_path: str):
        """初始化OCR任务
        
        Args:
            image_path: 图片路径
        """
        self.image_path = image_path
        self.logger = logging.getLogger(__name__)
        self.result = None
        self.status = "pending"
    
    def execute(self):
        """执行OCR任务"""
        try:
            self.status = "running"
            # TODO: 实现OCR识别逻辑
            self.status = "completed"
            return self
        except Exception as e:
            self.status = "failed"
            self.logger.error(f"OCR任务执行失败: {str(e)}")
            raise
    
    def get_text(self):
        """获取识别结果"""
        if self.status != "completed":
            raise RuntimeError("任务尚未完成")
        return self.result

class OCRTaskManager:
    """OCR任务管理器类"""
    
    def __init__(self):
        """初始化OCR任务管理器"""
        self.config = ConfigManager()
        self.logger = LogManager().get_logger("ocr")
        self.adb = ADBHelper()
        self.ocr = OCREngine()
        
        # 任务控制
        self.stop_event = Event()
        self.task_thread = None
        self.result_queue = Queue()
        
        # 创建GUI
        self.root = None
        self.status_label = None
        self.start_button = None
        self.stop_button = None
        self.result_text = None
    
    def setup_gui(self):
        """设置GUI界面"""
        self.root = tk.Tk()
        self.root.title("OCR任务控制")
        self.root.geometry("400x300")
        
        # 设置样式
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TLabel', padding=5)
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="任务状态: 未运行")
        self.status_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 控制按钮
        self.start_button = ttk.Button(
            main_frame,
            text="开始任务",
            command=self.start_task
        )
        self.start_button.grid(row=1, column=0, padx=5, pady=10)
        
        self.stop_button = ttk.Button(
            main_frame,
            text="停止任务",
            command=self.stop_task,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=1, column=1, padx=5, pady=10)
        
        # 结果文本框
        self.result_text = tk.Text(main_frame, height=10, width=40)
        self.result_text.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 设置定期检查任务状态
        self.root.after(1000, self.check_task_status)
    
    def check_task_status(self):
        """检查任务状态并更新界面"""
        if self.task_thread and self.task_thread.is_alive():
            self.status_label.config(text="任务状态: 运行中")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="任务状态: 未运行")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        
        # 更新结果文本框
        self.update_result_text()
        
        # 继续定期检查
        if self.root:
            self.root.after(1000, self.check_task_status)
    
    def update_result_text(self):
        """更新结果文本框"""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.result_text.insert(tk.END, f"{result}\n")
                self.result_text.see(tk.END)
        except:
            pass
    
    def start_task(self):
        """启动OCR任务"""
        try:
            if not self.task_thread or not self.task_thread.is_alive():
                self.stop_event.clear()
                self.task_thread = Thread(target=self._run_ocr_task)
                self.task_thread.daemon = True
                self.task_thread.start()
                
                self.logger.info("OCR任务已启动")
                messagebox.showinfo("成功", "OCR任务已启动")
        except Exception as e:
            self.logger.error(f"启动任务失败: {str(e)}")
            messagebox.showerror("错误", f"启动任务失败: {str(e)}")
    
    def stop_task(self):
        """停止OCR任务"""
        try:
            if self.task_thread and self.task_thread.is_alive():
                self.stop_event.set()
                self.task_thread.join(timeout=5)
                
                self.logger.info("OCR任务已停止")
                messagebox.showinfo("成功", "OCR任务已停止")
        except Exception as e:
            self.logger.error(f"停止任务失败: {str(e)}")
            messagebox.showerror("错误", f"停止任务失败: {str(e)}")
    
    def _run_ocr_task(self):
        """运行OCR任务的内部方法"""
        try:
            # 获取配置
            ocr_config = self.config.get('ocr')
            interval = ocr_config.get('interval', 5)  # 默认5秒
            
            self.logger.info(f"开始定时OCR任务，间隔时间：{interval}秒")
            
            while not self.stop_event.is_set():
                try:
                    # 执行截图
                    screenshot_path = self.adb.take_screenshot()
                    self.logger.info(f"截图成功：{screenshot_path}")
                    
                    # 执行OCR识别
                    result = self.ocr.recognize_image(screenshot_path)
                    if result['success']:
                        # 将识别结果放入队列
                        self.result_queue.put(f"识别结果: {result['text']}")
                        self.logger.info(f"OCR识别成功：{result['text']}")
                    else:
                        self.logger.error(f"OCR识别失败：{result.get('error', '未知错误')}")
                except Exception as e:
                    self.logger.error(f"任务执行失败：{str(e)}")
                
                # 等待指定时间或直到收到停止信号
                self.stop_event.wait(timeout=interval)
            
        except Exception as e:
            self.logger.error(f"任务异常：{str(e)}")
        finally:
            self.logger.info("OCR任务已结束")
    
    def run(self):
        """运行OCR任务管理器"""
        self.setup_gui()
        self.root.mainloop()
    
    def get_latest_result(self):
        """获取最新的识别结果"""
        try:
            return self.result_queue.get_nowait()
        except:
            return None

if __name__ == "__main__":
    manager = OCRTaskManager()
    manager.run() 