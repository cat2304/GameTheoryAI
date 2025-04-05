from flask import Flask, render_template, jsonify, request
import subprocess
import psutil
import sys
from pathlib import Path
import json

app = Flask(__name__)

# 任务进程
task_process = None

def get_task_status():
    """获取任务状态"""
    global task_process
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'screenshot_task.py' in ' '.join(proc.info['cmdline'] or []):
                task_process = proc
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    task_process = None
    return False

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """获取任务状态API"""
    is_running = get_task_status()
    return jsonify({
        'status': 'running' if is_running else 'stopped'
    })

@app.route('/api/start', methods=['POST'])
def start():
    """启动任务API"""
    try:
        global task_process
        if not get_task_status():
            script_path = Path(__file__).parent.parent / 'task' / 'screenshot_task.py'
            task_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return jsonify({'success': True, 'message': '任务已启动'})
        return jsonify({'success': False, 'message': '任务已在运行中'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop():
    """停止任务API"""
    try:
        global task_process
        if get_task_status() and task_process:
            task_process.terminate()
            task_process.wait(timeout=5)
            task_process = None
            return jsonify({'success': True, 'message': '任务已停止'})
        return jsonify({'success': False, 'message': '任务未在运行'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 