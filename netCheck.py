import requests
import socket
import subprocess
import platform
from urllib.parse import urlparse
import sys
import time
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


def check_proxy_settings():
    """检查当前的代理设置"""
    print("\n=== 代理设置检查 ===")
    proxy_settings = {
        'http_proxy': os.environ.get('http_proxy'),
        'https_proxy': os.environ.get('https_proxy'),
        'HTTP_PROXY': os.environ.get('HTTP_PROXY'),
        'HTTPS_PROXY': os.environ.get('HTTPS_PROXY')
    }

    for key, value in proxy_settings.items():
        print(f"{key}: {value}")


def test_connection(url, timeout=5):
    """测试与特定URL的连接"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        elapsed_time = time.time() - start_time
        print(f"✓ 成功连接到 {url}")
        print(f"  响应时间: {elapsed_time:.2f}秒")
        print(f"  状态码: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ 无法连接到 {url}")
        print(f"  错误: {str(e)}")
        return False


def ping_host(host):
    """Ping指定主机"""
    print(f"\n正在 Ping {host}...")

    # 根据操作系统选择合适的ping命令
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, '3', host]

    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"✓ 成功 Ping 通 {host}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ping {host} 失败")
        print(f"  错误: {e.stderr}")
        return False


def check_dns(host):
    """检查DNS解析"""
    print(f"\n检查 {host} 的DNS解析...")
    try:
        ip = socket.gethostbyname(host)
        print(f"✓ DNS解析成功: {host} -> {ip}")
        return True
    except socket.gaierror as e:
        print(f"✗ DNS解析失败: {host}")
        print(f"  错误: {str(e)}")
        return False


def main():
    # 检查代理设置
    check_proxy_settings()

    # 测试目标列表
    targets = [
        "https://huggingface.co",
        "https://cdn-lfs.huggingface.co",
        "https://datasets-server.huggingface.co",
        "https://raw.githubusercontent.com",
        "https://github.com"
    ]

    print("\n=== 连接测试 ===")
    for url in targets:
        print(f"\n测试连接 {url}")
        host = urlparse(url).netloc

        # DNS检查
        if check_dns(host):
            # Ping测试
            ping_host(host)
            # HTTP(S)连接测试
            test_connection(url)
        print("-" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        sys.exit(1)
