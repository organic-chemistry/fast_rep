import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots
import os
from fast_rep.read_data import load_bed_data

# Helper function to load data from BED files with column selection

    #return df["signal"],df[column], column_name


# Main plotting function
def plot_genomic_data(blocks, output_path, title="Genomic Data Visualization"):
    fig = make_subplots(rows=len(blocks), cols=1, shared_xaxes=True)
    for i, block in enumerate(blocks, 1):
        for signal in block:
            y_rounded = [float("%.2e" % v) if v != 0 else 0.0 for v in signal['y']]

            fig.add_trace(
                go.Scatter(
                    x=signal['x'],
                    y=y_rounded,
                    name=signal['name'],
                    mode='lines'
                ),
                row=i,
                col=1
            )
        fig.update_yaxes(title_text=f"Block {i}", row=i, col=1,tickformat=".2e")
    fig.update_layout(
        #height=800,
        #sizing= "stretch",
        title=title,
        showlegend=True,
        xaxis=dict(title="Genomic Position (kb)")
    )
    offline.plot(fig, filename=output_path, auto_open=False)

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Genomic Data Visualization Tool")
    parser.add_argument('--chromosome', '-c', type=str, required=True, help="Chromosome name (e.g., 'chr1')")
    parser.add_argument('--start', type=int, required=True, help="Start position")
    parser.add_argument('--end', type=int, required=True, help="End position")
    parser.add_argument('--resolution', type=float, default=5, help="Resolution in kilobases")
    parser.add_argument('--output', '-o', type=str, default="output.html", help="Output HTML file")
    parser.add_argument('--blocks', type=str, nargs='+', required=True,
                        help='Blocks of signals to plot. Format: '
                             '"SignalName,BEDfile.bed:column" or "BEDfile.bed:column" '
                             '(omit SignalName to use column name)')
    parser.add_argument('--nan0', action='store_true', help="Replace NaN with zeros")
    parser.add_argument('--nonan', action='store_true', help="Remove NaN values entirely")
    return parser.parse_args()

# Main execution
def main():
    args = parse_args()
    chromosome = args.chromosome
    start = args.start
    end = args.end
    resolution = args.resolution
    output_path = args.output
    blocks = []

    # Parse blocks
    for block_str in args.blocks:
        signals = block_str.split(',')
        block = []
        for entry in signals:
            # Split into name and file:column
            parts = entry.split(',', 1)
            if len(parts) == 2:
                name_part, file_col_part = parts
            else:
                name_part = None
                file_col_part = parts[0]

            # Split file and column
            try:
                file_part, column_part = file_col_part.split(':', 1)
            except ValueError:
                raise ValueError(f"Invalid format: {file_col_part}. Use 'file.bed:column'")

            # Load data and get column name
            try:
                x, y, column_name,meta = load_bed_data(
                    file_part, chromosome, start, end, resolution, column_part
                )
            except Exception as e:
                print(f"Error loading {file_part}: {e}")
                continue

            # Determine signal name
            if name_part is not None:
                signal_name = name_part
            else:
                signal_name = column_name

            # Apply NaN handling
            if args.nan0:
                y = np.nan_to_num(y)
            if args.nonan:
                valid = ~np.isnan(y)
                x, y = x[valid], y[valid]

            block.append({'x': x, 'y': y, 'name': signal_name})
        blocks.append(block)

    # Plot
    plot_genomic_data(blocks, output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()