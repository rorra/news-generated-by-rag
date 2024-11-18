"""
Results Visualization Script

This script generates visualizations and reports from RAG evaluation results.
"""

import argparse
from pathlib import Path
from evaluation.visualization import ResultsVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for RAG evaluation results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save visualizations and report'
    )
    parser.add_argument(
        '--format',
        choices=['html', 'png', 'all'],
        default='all',
        help='Output format for visualizations'
    )

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = ResultsVisualizer(args.results_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.format in ['html', 'all']:
        print("Generating HTML report...")
        visualizer.generate_report(output_dir)

    if args.format in ['png', 'all']:
        print("Generating individual plots...")
        # Generate individual plots
        metrics_fig = visualizer.create_metrics_comparison()
        metrics_fig.write_image(output_dir / 'metrics_comparison.png')

        section_fig = visualizer.create_section_performance()
        section_fig.write_image(output_dir / 'section_performance.png')

        query_fig = visualizer.create_query_type_comparison()
        query_fig.write_image(output_dir / 'query_distribution.png')

        time_fig = visualizer.create_execution_time_plot()
        time_fig.write_image(output_dir / 'execution_time.png')

    print(f"Visualizations generated in {output_dir}")


if __name__ == '__main__':
    main()
