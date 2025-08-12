import pandas as pd
import io
import webbrowser
import os
from datetime import datetime

def analyze_performance_data(df):
    valid_runs = df[df['tokens_per_second_gen'] > 0].copy()
    if valid_runs.empty:
        return None, df
    best_run = valid_runs.loc[valid_runs['tokens_per_second_gen'].idxmax()]
    return best_run, valid_runs

def create_html_report(best_run, all_runs_df, llamacpp_path, model_dir_path):
    if best_run is not None:
        model_name = best_run['model_filename']
        quant_format = best_run['Format']
        gen_speed = best_run['tokens_per_second_gen']
        compat_score = best_run['Compat']
        cpu_name = best_run['cpu']
        ram_gb = best_run['ram_total_gb']
        gpu_name = best_run['gpu']
        vram_gb = best_run['gpu_vram_gb']

        analysis_html = f"""
        <div class="card">
            <h2>Performance Highlights</h2>
            <p>Based on your test data, your system achieves its best performance with the <b>{model_name}</b> model using <b>{quant_format}</b> quantization.</p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Generation Speed</h4>
                    <p class="metric-value">{gen_speed:.2f}</p>
                    <p class="metric-unit">tokens/second</p>
                </div>
                <div class="metric-card">
                    <h4>Compatibility Score</h4>
                    <p class="metric-value">{compat_score}</p>
                    <p class="metric-unit">System Rating</p>
                </div>
            </div>
            <h4>System Configuration for Best Run:</h4>
            <ul>
                <li><b>CPU:</b> {cpu_name}</li>
                <li><b>GPU:</b> {gpu_name if gpu_name != 'unknown' else f'Not named (VRAM: {vram_gb} GB)'}</li>
                <li><b>System RAM:</b> {ram_gb} GB</li>
            </ul>
        </div>
        """
    else:
        analysis_html = """
        <div class="card">
            <h2>Performance Highlights</h2>
            <p>No successful performance runs were found in the provided data. Cannot generate a run command.</p>
        </div>
        """

    run_command_html = ""
    if best_run is not None and llamacpp_path and model_dir_path:
        model_path = os.path.join(model_dir_path, best_run['model_filename'])
        gpu_layers = "-ngl 35" if best_run['gpu_vram_gb'] > 0 else ""
        command = f"{llamacpp_path} -m \"{model_path}\" {gpu_layers} -n -1 --color -i -c 2048"
        
        run_command_html = f"""
        <div class="card">
            <h2>How to Run the Best Model</h2>
            <p>Use the following command in your terminal to run the best-performing model (<b>{best_run['model_filename']}</b>) in an interactive chat session.</p>
            <div class="command-box">
                <pre><code>{command}</code></pre>
                <button class="copy-btn" onclick="copyCommand(this)">Copy</button>
            </div>
        </div>
        """

    table_rows = []
    def get_compat_style(value):
        if 'GOOD' in str(value):
            return 'style="color: #1e8e3e; font-weight: bold;"'
        elif 'Warning' in str(value):
            return 'style="color: #e8710a; font-weight: bold;"'
        return ''

    for _, row in all_runs_df.iterrows():
        style = get_compat_style(row['Compat'])
        table_rows.append(f"""
        <tr>
            <td>{row['model_filename']}</td>
            <td>{row['Format']}</td>
            <td>{row['tokens_per_second_gen']:.2f}</td>
            <td>{row['avg_cpu_usage_percent']:.1f}%</td>
            <td>{row['avg_gpu_usage_percent']:.1f}%</td>
            <td>{row['peak_vram_usage_mb']:.2f}</td>
            <td {style}>{row['Compat']}</td>
        </tr>
        """)

    table_html = f"""
    <div class="card">
        <h2>Detailed Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model Filename</th>
                    <th>Quantization</th>
                    <th>Gen Speed (tok/s)</th>
                    <th>Avg CPU Usage</th>
                    <th>Avg GPU Usage</th>
                    <th>Peak VRAM (MB)</th>
                    <th>Compatibility</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>LLM Performance Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; }}
            h1 {{ text-align: center; color: #1a73e8; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-bottom: 10px; }}
            h1 .subtitle {{ display: block; font-size: 0.6em; color: #666; font-weight: normal; margin-top: 5px; }}
            h2 {{ color: #202124; border-bottom: 1px solid #e0e0e0; padding-bottom: 5px; margin-top: 0; }}
            .card {{ background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
            th {{ font-weight: 600; color: #5f6368; background-color: #f8f9fa; }}
            tr:hover {{ background-color: #f1f3f4; }}
            ul {{ padding-left: 20px; }}
            .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .metric-card {{ background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #1a73e8; margin: 0; }}
            .metric-unit {{ font-size: 0.9em; color: #5f6368; margin: 0; }}
            .command-box {{ background-color: #202124; color: #e8eaed; font-family: "Courier New", Courier, monospace; padding: 15px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; }}
            .command-box pre {{ margin: 0; white-space: pre-wrap; word-break: break-all; }}
            .copy-btn {{ background-color: #4285f4; color: white; border: none; border-radius: 4px; padding: 8px 12px; cursor: pointer; }}
            .copy-btn:hover {{ background-color: #1a73e8; }}
        </style>
        <script>
            function copyCommand(button) {{
                const pre = button.previousElementSibling;
                const command = pre.textContent;
                navigator.clipboard.writeText(command).then(() => {{
                    button.textContent = 'Copied!';
                    setTimeout(() => {{ button.textContent = 'Copy'; }}, 2000);
                }});
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>LLM Performance Report <span class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></h1>
            {analysis_html}
            {run_command_html}
            {table_html}
        </div>
    </body>
    </html>
    """
    return html_content

def pmain(csv_path, llamacpp_path, model_dir_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    
    best_run, _ = analyze_performance_data(df)
    
    html_output = create_html_report(best_run, df, llamacpp_path, model_dir_path)

    report_filename = "llm_performance_report.html"
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(html_output)
        
        filepath = os.path.abspath(report_filename)
        print(f"\nReport generated successfully: {filepath}")
        
        webbrowser.open(f"file://{filepath}")

    except IOError as e:
        print(f"\nError: Could not write report to file. {e}")
