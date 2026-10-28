import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def parse_summary_markdown(filepath='summary.md'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found in the current directory.")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    datasets = {}
    current_dataset = None
    headers = []
    rows = []

    for line in lines:
        if line.startswith('## Dataset:'):
            if current_dataset and rows:
                datasets[current_dataset] = {'headers': headers, 'rows': rows}
            current_dataset = line.split(':', 1)[1].strip()
            headers = []
            rows = []
        elif line.startswith('|') and '---' in line:
            continue
        elif line.startswith('|'):
            cols = [c.strip() for c in line.split('|')[1:-1]]
            if not headers:
                headers = cols
            else:
                rows.append(cols)

    if current_dataset and rows:
        datasets[current_dataset] = {'headers': headers, 'rows': rows}

    all_rows = []
    for ds, data in datasets.items():
        headers = data['headers']
        for r in data['rows']:
            row_dict = {}
            for h, val in zip(headers, r):
                if h in ['scale', 'noise_sigma', 'n_images', 'param_count']:
                    row_dict[h] = int(val) if val != '' else 0
                elif h in ['mean_inference_time_s', 'mean_psnr', 'mean_ssim']:
                    row_dict[h] = float(val) if val != '' else 0.0
                else:
                    row_dict[h] = val
            all_rows.append(row_dict)

    return pd.DataFrame(all_rows)

def generate_advanced_dashboard(df, output_html='super_resolution_dashboard.html'):
    template_theme = "plotly_white"

    # 1. Bubble Scatter: PSNR vs Inference Time
    fig_scatter = px.scatter(
        df,
        x="mean_inference_time_s",
        y="mean_psnr",
        color="model",
        symbol="dataset",
        size="param_count",
        hover_data=["experiment", "scale", "noise_sigma", "mean_ssim", "degradation"],
        title="<b>1. Performance vs. Speed Trade-off</b> (PSNR vs. Inference Time)",
        labels={
            "mean_inference_time_s": "Mean Inference Time (s)",
            "mean_psnr": "Mean PSNR (dB)",
            "model": "Model Family",
            "dataset": "Benchmark Dataset"
        }
    )
    fig_scatter.update_layout(template=template_theme, hovermode="closest", height=550)

    # 2. Grouped Bar Chart: Noise & Scale Robustness
    fig_bar = px.bar(
        df,
        x="dataset",
        y="mean_psnr",
        color="model",
        barmode="group",
        facet_col="scale",
        facet_row="noise_sigma",
        title=r"<b>2. Robustness Analysis Across Scales & Noise Levels ($\sigma$)</b>",
        labels={"mean_psnr": "Mean PSNR (dB)", "dataset": "Dataset", "model": "Model"}
    )
    fig_bar.update_layout(template=template_theme, height=700)

    # 3. Box Plot: Structural Similarity (SSIM) Distribution across Models
    fig_violin = px.box(
        df,
        x="model",
        y="mean_ssim",
        color="model",
        points="all",
        title="<b>3. SSIM Quality Distribution per Model Architecture</b>",
        labels={"mean_ssim": "Mean SSIM", "model": "Model"}
    )
    fig_violin.update_layout(template=template_theme, showlegend=False, height=550)

    # 4. Parameter Count vs SSIM Efficiency Frontier (Optimized to prevent label overlap)
    fig_eff = px.scatter(
        df,
        x="param_count",
        y="mean_ssim",
        color="model",
        symbol="dataset",
        hover_data=["experiment", "scale", "noise_sigma", "mean_psnr"],
        title="<b>4. Parameter Efficiency Frontier</b> (SSIM vs. Model Parameters)",
        labels={"param_count": "Parameter Count", "mean_ssim": "Mean SSIM"}
    )
    # Use cleaner markers and cleaner hover setup instead of crowding with text labels
    fig_eff.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
    fig_eff.update_layout(
        template=template_theme, 
        height=600,
        hovermode="closest"
    )

    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Super-Resolution Benchmark Dynamic Dashboard (#10)</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 30px;
                background-color: #f4f6f9;
                color: #333;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
            }}
            .header h1 {{
                color: #1e293b;
                font-size: 2.2rem;
                margin-bottom: 10px;
            }}
            .header p {{
                color: #64748b;
                font-size: 1.1rem;
            }}
            .grid-container {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 30px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            .card {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Super-Resolution Dynamic Comparison Dashboard</h1>
            <p>GitHub Issue #10 Solution — Multi-view interactive analysis generated from <code>summary.md</code></p>
        </div>
        
        <div class="grid-container">
            <div class="card">
                {fig_scatter.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            <div class="card">
                {fig_bar.to_html(full_html=False, include_plotlyjs='false')}
            </div>
            <div class="card">
                {fig_violin.to_html(full_html=False, include_plotlyjs='false')}
            </div>
            <div class="card">
                {fig_eff.to_html(full_html=False, include_plotlyjs='false')}
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"Successfully generated clean multi-view dynamic dashboard: {output_html}")

if __name__ == '__main__':
    df = parse_summary_markdown('summary.md')
    generate_advanced_dashboard(df)