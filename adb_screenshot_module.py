# adb_screenshot_module.py

"""
ADB 截图模块

该模块负责通过 ADB (Android Debug Bridge) 从连接的安卓设备或模拟器截取屏幕图像。
"""

import subprocess
import os
import time

class ADBScreenshot:
    def __init__(self, adb_path="adb", device_id=None):
        """
        初始化 ADB截图模块。
        adb_path (str): ADB 可执行文件的路径。如果 ADB 在系统 PATH 中，则保留默认值 "adb"。
        device_id (str, optional): 目标设备的序列号。如果只有一个设备/模拟器连接，则可以为 None。
        """
        self.adb_path = adb_path
        self.device_id_arg = ["-s", device_id] if device_id else []
        self._test_adb_connection()

    def _test_adb_connection(self):
        """
        测试 ADB 是否能正常连接到设备。
        """
        try:
            command = [self.adb_path] + self.device_id_arg + ["devices"]
            print(f"测试 ADB 连接: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
            if self.device_id_arg:
                if self.device_id_arg[1] not in process.stdout:
                    raise Exception(f"设备 {self.device_id_arg[1]} 未找到或未授权。检测到的设备: \n{process.stdout}")
                print(f"ADB 连接到设备 {self.device_id_arg[1]} 成功。")
            elif "device" not in process.stdout:
                 # 如果没有指定device_id，并且输出中没有device，可能表示没有设备或模拟器连接
                 if "emulator" not in process.stdout.lower() and "offline" not in process.stdout.lower(): # 排除 offline 状态
                    print(f"警告: 未检测到活动设备/模拟器，或设备未授权。ADB devices 输出: \n{process.stdout}")
                 else:
                    print(f"ADB 服务运行正常，检测到以下设备/模拟器:\n{process.stdout}")        
            else:
                print(f"ADB 连接成功。检测到以下设备/模拟器:\n{process.stdout}")

        except subprocess.CalledProcessError as e:
            print(f"ADB 命令执行失败: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise Exception("ADB 命令执行失败，请检查 ADB 环境和设备连接。")
        except subprocess.TimeoutExpired:
            print("ADB 命令超时。")
            raise Exception("ADB 命令超时，请检查设备连接状态。")
        except FileNotFoundError:
            print(f"错误: ADB 可执行文件未在路径 '{self.adb_path}' 找到。请确保 ADB 已安装并配置在系统 PATH 中，或提供正确的路径。")
            raise

    def capture_screenshot(self, remote_path="/sdcard/screen.png", local_dir="/home/ubuntu/screenshots", local_filename=None):
        """
        截取屏幕图像并将其保存到本地。

        remote_path (str): 图像在设备上临时保存的路径。
        local_dir (str): 本地保存截图的目录。
        local_filename (str, optional): 本地保存截图的文件名。如果为 None，则使用时间戳生成文件名。

        Returns:
            str: 本地截图文件的完整路径，如果成功则返回路径，否则返回 None。
        """
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"创建本地截图目录: {local_dir}")

        if local_filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            local_filename = f"screenshot_{timestamp}.png"
        
        local_filepath = os.path.join(local_dir, local_filename)

        try:
            # 1. 在设备上截图
            screencap_command = [self.adb_path] + self.device_id_arg + ["shell", "screencap", "-p", remote_path]
            print(f"执行截图命令: {' '.join(screencap_command)}")
            subprocess.run(screencap_command, check=True, timeout=15)
            print(f"截图已保存到设备: {remote_path}")

            # 2. 将截图从设备拉取到本地
            pull_command = [self.adb_path] + self.device_id_arg + ["pull", remote_path, local_filepath]
            print(f"执行拉取命令: {' '.join(pull_command)}")
            subprocess.run(pull_command, check=True, timeout=15)
            print(f"截图已成功拉取到本地: {local_filepath}")

            # 3. (可选) 删除设备上的临时截图文件
            rm_command = [self.adb_path] + self.device_id_arg + ["shell", "rm", remote_path]
            print(f"执行删除设备截图命令: {' '.join(rm_command)}")
            subprocess.run(rm_command, check=False, timeout=10) # check=False 因为即使删除失败，截图也已获取
            print(f"已尝试删除设备上的临时截图: {remote_path}")
            
            return local_filepath

        except subprocess.CalledProcessError as e:
            print(f"ADB 命令执行失败: {e}")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return None
        except subprocess.TimeoutExpired as e:
            print(f"ADB 命令超时: {' '.join(e.cmd)}")
            return None
        except Exception as e:
            print(f"发生未知错误: {e}")
            return None

if __name__ == '__main__':
    print("测试 ADB Screenshot 模块...")
    # 假设 MuMu 模拟器通常在 127.0.0.1:7555 (需要先用 adb connect 连接)
    # 或者如果 ADB server 已经知道模拟器，则不需要 device_id
    # adb connect 127.0.0.1:7555 (如果尚未连接)
    
    # 尝试连接到模拟器 (如果需要)
    # try:
    #     subprocess.run(["adb", "connect", "127.0.0.1:7555"], timeout=5, check=True, capture_output=True)
    #     print("尝试连接到 127.0.0.1:7555 成功或已连接。")
    # except Exception as e:
    #     print(f"连接到 127.0.0.1:7555 失败 (可能是已连接或模拟器未运行): {e}")

    # 使用默认的 ADB 路径，不指定 device_id (适用于单个连接的模拟器)
    # 如果有多个设备，需要指定 device_id="emulator-xxxx" 或 "127.0.0.1:7555"
    # screenshotter = ADBScreenshot(device_id="127.0.0.1:7555") # 示例：如果MuMu模拟器是这个ID
    screenshotter = ADBScreenshot() # 适用于单个已连接的模拟器
    
    # 创建截图保存目录
    save_directory = "/home/ubuntu/mumu_screenshots"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    print(f"\n开始进行截图测试，截图将保存在 {save_directory} 目录中。")
    filepath = screenshotter.capture_screenshot(local_dir=save_directory)

    if filepath and os.path.exists(filepath):
        print(f"\n截图测试成功！文件保存在: {filepath}")
        print(f"文件大小: {os.path.getsize(filepath)} bytes")
    else:
        print("\n截图测试失败。请检查 ADB 连接、模拟器状态和权限设置。")

    # 示例：指定设备ID (如果知道模拟器的ID，例如 'emulator-5554' 或 '127.0.0.1:7555')
    # print("\n尝试使用特定设备ID进行截图...")
    # specific_device_id = "127.0.0.1:7555" # 替换为你的模拟器ID
    # try:
    #    screenshotter_specific = ADBScreenshot(device_id=specific_device_id)
    #    filepath_specific = screenshotter_specific.capture_screenshot(local_dir=save_directory, local_filename="specific_device_shot.png")
    #    if filepath_specific and os.path.exists(filepath_specific):
    #        print(f"特定设备截图成功！文件保存在: {filepath_specific}")
    #    else:
    #        print("特定设备截图失败。")
    # except Exception as e:
    #    print(f"初始化特定设备截图器失败: {e}")

