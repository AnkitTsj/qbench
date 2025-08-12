import psutil
import platform
import socket
import os
import sys
import subprocess
import webbrowser
from datetime import datetime

try:
    import cpuinfo
except ImportError:
    cpuinfo = None
    print("Warning: 'py-cpuinfo' not installed. CPU brand and frequency will be missing.")
    print("Install with: pip install py-cpuinfo")

try:
    import GPUtil
except ImportError:
    GPUtil = None
    print("Warning: 'GPUtil' not installed. Falling back to nvidia-smi if available.")
    print("Install with: pip install gputil")

def get_os_info():
    return {
        'OS': platform.system(),
        'OS Version': platform.version(),
        'Architecture': platform.architecture()[0],
        'Machine': platform.machine(),
        'Processor': platform.processor()
    }

def get_cpu_info():
    info = {}
    if cpuinfo:
        cpu_details = cpuinfo.get_cpu_info()
        info['Brand'] = cpu_details.get('brand_raw', 'N/A')
        info['Cores (Physical)'] = psutil.cpu_count(logical=False)
        info['Threads (Logical)'] = psutil.cpu_count(logical=True)
        info['Frequency'] = cpu_details.get('hz_actual_friendly', 'N/A')
    else:
        info['Brand'] = 'N/A (py-cpuinfo not installed)'
        info['Cores (Physical)'] = psutil.cpu_count(logical=False)
        info['Threads (Logical)'] = psutil.cpu_count(logical=True)
        info['Frequency'] = 'N/A'
    return info

def get_memory_info():
    mem = psutil.virtual_memory()
    return {
        'Total RAM': f"{mem.total / (1024**3):.2f} GB",
        'Available RAM': f"{mem.available / (1024**3):.2f} GB",
        'Used RAM': f"{mem.used / (1024**3):.2f} GB",
        'Percent Used': mem.percent
    }

def get_storage_info():
    partitions = psutil.disk_partitions()
    storage = {}
    for partition in partitions:
        if 'loop' in partition.device or partition.fstype == '' or not os.path.exists(partition.mountpoint):
            continue
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            storage[partition.mountpoint] = {
                'Device': partition.device,
                'Total': f"{usage.total / (1024**3):.2f} GB",
                'Used': f"{usage.used / (1024**3):.2f} GB",
                'Free': f"{usage.free / (1024**3):.2f} GB",
                'Percent Used': usage.percent
            }
        except (PermissionError, FileNotFoundError):
            continue
    return storage

def get_gpu_info():
    gpus = []
    if GPUtil:
        gpu_list = GPUtil.getGPUs()
        for gpu in gpu_list:
            gpus.append({'ID': gpu.id, 'Name': gpu.name, 'Total VRAM': f"{gpu.memoryTotal / 1024:.2f} GB", 'Free VRAM': f"{gpu.memoryFree / 1024:.2f} GB", 'Used VRAM': f"{gpu.memoryUsed / 1024:.2f} GB", 'Load': f"{gpu.load * 100:.1f}%", 'Temperature': f"{gpu.temperature} °C"})
        return gpus
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], encoding='utf-8')
        lines = output.strip().split('\n')
        for line in lines:
            if not line: continue
            parts = [p.strip() for p in line.split(',')]
            gpus.append({'ID': parts[0], 'Name': parts[1], 'Total VRAM': f"{float(parts[2]) / 1024:.2f} GB", 'Free VRAM': f"{float(parts[3]) / 1024:.2f} GB", 'Used VRAM': f"{float(parts[4]) / 1024:.2f} GB", 'Load': f"{parts[5]}%", 'Temperature': f"{parts[6]} °C"})
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return gpus

def get_network_info():
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        ip_address = "127.0.0.1 (No network connection)"
    return {'Hostname': socket.gethostname(), 'IP Address': ip_address}


def estimate_llm_capacity(ram_gb, vram_gb=0, safe_buffer=0.85):
    effective_mem_gb = vram_gb if vram_gb > 1 else ram_gb
    resource_type = "GPU VRAM" if vram_gb > 1 else "System RAM"
    
    available_mem_gb = effective_mem_gb * safe_buffer
    estimated_model_size_gb = available_mem_gb 
    
    if available_mem_gb < 1:
        return "Not enough free memory detected for running most LLMs.", 0

    max_params_fp16 = available_mem_gb / 2      
    max_params_8bit = available_mem_gb / 1      
    max_params_6bit = available_mem_gb / 0.75
    max_params_5bit = available_mem_gb / 0.625  
    max_params_4bit = available_mem_gb / 0.5    
    
    result_text = (
        f"Based on {available_mem_gb:.1f} GB of available {resource_type} (using an {safe_buffer*100:.0f}% buffer):\n\n"
        f"- FP16 (no quantization): Can run models up to ~{max_params_fp16:.1f} Billion parameters.\n"
        f"  (e.g., Llama-3-8B needs ~16GB)\n\n"
        f"- 8-Bit Quantized:        Can run models up to ~{max_params_8bit:.1f} Billion parameters.\n"
        f"  (e.g., Llama-3-8B needs ~8GB)\n\n"
        f"- 6-Bit Quantized:        Can run models up to ~{max_params_6bit:.1f} Billion parameters.\n\n"
        f"- 5-Bit Quantized:        Can run models up to ~{max_params_5bit:.1f} Billion parameters.\n\n"
        f"- 4-Bit Quantized:        Can run models up to ~{max_params_4bit:.1f} Billion parameters.\n"
        f"  (e.g., Llama-3-8B needs ~4-5GB)\n\n"
        f"Note: These are estimates. Actual memory usage depends on the model, context length, and batch size."
    )
    return result_text, estimated_model_size_gb

# --- Storage Check Function (unchanged) ---
def check_storage_for_llm(storage_info, model_size_gb, support_files_gb=2):
    if model_size_gb == 0:
        return "Not applicable, as no sufficient memory for an LLM was found."

    required_space_gb = model_size_gb + support_files_gb
    suitable_drives = []
    max_free_space = 0
    drive_with_max_space = ""

    for mountpoint, info in storage_info.items():
        try:
            free_gb = float(info['Free'].split()[0])
            if free_gb > max_free_space:
                max_free_space = free_gb
                drive_with_max_space = mountpoint
            if free_gb >= required_space_gb:
                suitable_drives.append(f"<b>{mountpoint}</b>")
        except (ValueError, IndexError):
            continue

    if suitable_drives:
        return f" <b>Ready:</b> Drive(s) {', '.join(suitable_drives)} have enough space for an estimated <b>{model_size_gb:.1f} GB</b> model plus {support_files_gb} GB of supporting files."
    elif drive_with_max_space:
        shortfall = required_space_gb - max_free_space
        if shortfall > 0:
            return f" <b>Action Required:</b> No single drive has enough space. To store the estimated <b>{model_size_gb:.1f} GB</b> model, you need to free up approximately <b>{shortfall:.1f} GB</b> on drive '<b>{drive_with_max_space}</b>'."
        else: 
            return f" <b>Ready:</b> Drive <b>{drive_with_max_space}</b> has just enough space for the <b>{model_size_gb:.1f} GB</b> model."
    else:
        return "Could not assess storage for any suitable drives."

def generate_html_report(data):
    def create_progress_bar(percent):
        return f'<div class="progress-container"><div class="progress-bar" style="width: {percent}%;">{percent}%</div></div>'
    def create_table_rows(info_dict):
        return ''.join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in info_dict.items())

    storage_html = ""
    if data['storage']:
        for mount, info in data['storage'].items():
            storage_html += f"""<div class="card"><h2>Storage: {mount} ({info['Device']})</h2><table><tr><th>Total Size</th><td>{info['Total']}</td></tr><tr><th>Used Space</th><td>{info['Used']}</td></tr><tr><th>Free Space</th><td>{info['Free']}</td></tr><tr><th>Usage</th><td>{create_progress_bar(info['Percent Used'])}</td></tr></table></div>"""
    else:
        storage_html = "<div class='card'><h2>Storage</h2><p>No storage devices found or accessible.</p></div>"
    gpu_html = ""
    if data['gpus']:
        for gpu in data['gpus']:
            gpu_html += f"""<div class="card"><h2>GPU {gpu['ID']}: {gpu['Name']}</h2><table>{create_table_rows({k: v for k, v in gpu.items() if k not in ['ID', 'Name']})}</table></div>"""
    else:
        gpu_html = "<div class='card'><h2>GPU</h2><p>No compatible GPU detected.</p></div>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>System Hardware Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: auto; }}
            h1 {{ text-align: center; color: #1a73e8; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-bottom: 10px; }}
            h1 .subtitle {{ display: block; font-size: 0.6em; color: #666; font-weight: normal; margin-top: 5px; }}
            h2 {{ color: #202124; border-bottom: 1px solid #e0e0e0; padding-bottom: 5px; margin-top: 0; }}
            .card {{ background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
            th {{ font-weight: 600; width: 35%; color: #5f6368; }}
            tr:last-child th, tr:last-child td {{ border-bottom: none; }}
            pre {{ background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 15px; white-space: pre-wrap; word-wrap: break-word; font-family: "Courier New", Courier, monospace; font-size: 14px; line-height: 1.6; }}
            .progress-container {{ background-color: #e0e0e0; border-radius: 4px; width: 100%; }}
            .progress-bar {{ background-color: #4285f4; color: white; padding: 2px 5px; text-align: center; font-size: 12px; border-radius: 4px; line-height: 1.5; }}
            .readiness-check {{ background-color: #e8f0fe; border-left: 5px solid #1a73e8; padding: 15px; margin-top: -5px; margin-bottom: 20px; }}
            .readiness-check p {{ margin: 0; font-size: 1.1em; color: #202124; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>System Hardware Report <span class="subtitle">Generated on {data['timestamp']}</span></h1>
            
            <div class="card">
                <h2>LLM Capacity Estimation</h2>
                <pre>{data['llm_capacity']}</pre>
            </div>

            <div class="readiness-check">
                <p>{data['storage_readiness']}</p>
            </div>

            <div class="card"><h2>Operating System</h2><table>{create_table_rows(data['os'])}</table></div>
            <div class="card"><h2>CPU (Central Processing Unit)</h2><table>{create_table_rows(data['cpu'])}</table></div>
            <div class="card"><h2>Memory (RAM)</h2><table><tr><th>Total RAM</th><td>{data['memory']['Total RAM']}</td></tr><tr><th>Available RAM</th><td>{data['memory']['Available RAM']}</td></tr><tr><th>Used RAM</th><td>{data['memory']['Used RAM']}</td></tr><tr><th>Usage</th><td>{create_progress_bar(data['memory']['Percent Used'])}</td></tr></table></div>
            {gpu_html}
            {storage_html}
            <div class="card"><h2>Network</h2><table>{create_table_rows(data['network'])}</table></div>
        </div>
    </body>
    </html>
    """
    return html_content

def main():
    print("Gathering system information...")
    
    os_info = get_os_info()
    cpu_info = get_cpu_info()
    mem_info = get_memory_info()
    storage_info = get_storage_info()
    gpu_info = get_gpu_info()
    network_info = get_network_info()

    available_ram_gb = float(mem_info['Available RAM'].split()[0])
    free_vram_gb = 0
    if gpu_info:
        for gpu in gpu_info:
            try:
                free_vram_gb += float(gpu['Free VRAM'].split()[0])
            except (ValueError, IndexError):
                pass
    
    llm_capacity_text, est_model_size_gb = estimate_llm_capacity(available_ram_gb, free_vram_gb)
    
    storage_readiness_msg = check_storage_for_llm(storage_info, est_model_size_gb)
    
    report_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'os': os_info,
        'cpu': cpu_info,
        'memory': mem_info,
        'storage': storage_info,
        'gpus': gpu_info,
        'network': network_info,
        'llm_capacity': llm_capacity_text,
        'storage_readiness': storage_readiness_msg,
    }

    html_output = generate_html_report(report_data)

    report_filename = "system_report.html"
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(html_output)
        filepath = os.path.abspath(report_filename)
        print(f"\nReport generated successfully: {filepath}")
        webbrowser.open(f"file://{filepath}")
    except IOError as e:
        print(f"\nError: Could not write report to file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()