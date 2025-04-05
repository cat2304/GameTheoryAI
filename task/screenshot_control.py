import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import psutil
import os
import sys
from pathlib import Path

class ScreenshotControl:
    def __init__(self, root):
        self.root = root
        self.root.title("截图任务控制")
        self.root.geometry("300x200")
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure('TButton', padding=5)
        self.style.configure('TLabel', padding=5)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 状态标签
        self.status_label = ttk.Label(self.main_frame, text="任务状态: 未运行")
        self.status_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 控制按钮
        self.start_button = ttk.Button(
            self.main_frame, 
            text="开始任务", 
            command=self.start_task
        )
        self.start_button.grid(row=1, column=0, padx=5, pady=10)
        
        self.stop_button = ttk.Button(
            self.main_frame, 
            text="停止任务", 
            command=self.stop_task,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=1, column=1, padx=5, pady=10)
        
        # 任务进程
        self.task_process = None
        
        # 检查任务状态
        self.check_task_status()
    
    def check_task_status(self):
        """检查任务是否在运行"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'screenshot_task.py' in ' '.join(proc.info['cmdline'] or []):
                    self.task_process = proc
                    self.status_label.config(text="任务状态: 运行中")
                    self.start_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.NORMAL)
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        self.task_process = None
        self.status_label.config(text="任务状态: 未运行")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def start_task(self):
        """启动截图任务"""
        try:
            # 获取脚本路径
            script_path = Path(__file__).parent / 'screenshot_task.py'
            
            # 启动任务
            self.task_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 更新界面状态
            self.status_label.config(text="任务状态: 运行中")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            messagebox.showinfo("成功", "截图任务已启动")
        except Exception as e:
            messagebox.showerror("错误", f"启动任务失败: {str(e)}")
    
    def stop_task(self):
        """停止截图任务"""
        try:
            if self.task_process:
                # 终止进程
                self.task_process.terminate()
                self.task_process.wait(timeout=5)
                
                # 更新界面状态
                self.status_label.config(text="任务状态: 未运行")
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                
                messagebox.showinfo("成功", "截图任务已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止任务失败: {str(e)}")
    
    def run(self):
        """运行界面"""
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenshotControl(root)
    app.run() 