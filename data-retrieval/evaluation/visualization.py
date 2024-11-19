"""
Visualization Utilities for RAG Evaluation Results

This module provides functions for visualizing and comparing the performance
of different embedding strategies using various metrics and charts.
"""

from typing import List, Dict, Any
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


class ResultsVisualizer:
    """Class for creating visualizations of RAG evaluation results."""

    def __init__(self, results_dir: str):
        """Initialize the visualizer with results directory."""
        self.results_dir = Path(results_dir)
        self.embedders = []
        self.results = {}
        self._load_results()

    def _load_results(self):
        """Load all JSON result files from the results directory."""
        for file_path in self.results_dir.glob('*_report.json'):
            embedder = file_path.stem.replace('_report', '')
            with open(file_path) as f:
                self.results[embedder] = json.load(f)
                if embedder not in self.embedders:
                    self.embedders.append(embedder)

    def create_metrics_comparison(self) -> go.Figure:
        """Create a bar chart comparing key metrics across embedders."""
        df_metrics = []

        for embedder, data in self.results.items():
            metrics = data['metrics']
            df_metrics.append({
                'Embedder': embedder,
                'Semantic Precision': metrics['precision_at_k'],
                'Semantic Recall': metrics['recall_at_k'],
                'NDCG': metrics['ndcg'],
                'Keyword Precision': metrics['keyword_precision'],
                'Keyword Recall': metrics['keyword_recall'],
                'Keyword F1': metrics['keyword_f1'],
                'QPS': metrics['queries_per_second']
            })

        df = pd.DataFrame(df_metrics)

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Semantic Retrieval Metrics',
                'Keyword-based Metrics',
                'Performance (Queries/Second)'
            ),
            vertical_spacing=0.2,
            row_heights=[0.4, 0.4, 0.2]
        )

        # Semantic metrics
        for metric in ['Semantic Precision', 'Semantic Recall', 'NDCG']:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=df['Embedder'],
                    y=df[metric],
                    text=df[metric].round(3),
                    textposition='auto',
                ),
                row=1, col=1
            )

        # Keyword metrics
        for metric in ['Keyword Precision', 'Keyword Recall', 'Keyword F1']:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=df['Embedder'],
                    y=df[metric],
                    text=df[metric].round(3),
                    textposition='auto',
                ),
                row=2, col=1
            )

        # Performance metric
        fig.add_trace(
            go.Bar(
                name='QPS',
                x=df['Embedder'],
                y=df['QPS'],
                text=df['QPS'].round(2),
                textposition='auto',
            ),
            row=3, col=1
        )

        fig.update_layout(
            title_text='Embedding Strategies Comparison',
            height=1000,
            showlegend=True,
            barmode='group'
        )

        return fig

    def create_query_type_performance(self) -> go.Figure:
        """Create a heatmap showing performance across different query types."""
        performance_data = []

        for embedder, data in self.results.items():
            metrics = data['metrics']
            performance_data.append({
                'Embedder': embedder,
                'Semantic Only': metrics['precision_at_k'],
                'Keyword Only': metrics['keyword_precision'],
                'Combined': (metrics['precision_at_k'] + metrics['keyword_precision']) / 2,
            })

        df = pd.DataFrame(performance_data)

        fig = go.Figure(data=go.Heatmap(
            z=df[['Semantic Only', 'Keyword Only', 'Combined']].values,
            x=['Semantic Only', 'Keyword Only', 'Combined'],
            y=df['Embedder'],
            text=df[['Semantic Only', 'Keyword Only', 'Combined']].round(3).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdYlGn'
        ))

        fig.update_layout(
            title='Query Type Performance by Embedder',
            xaxis_title='Query Type',
            yaxis_title='Embedder'
        )

        return fig

    def create_execution_time_plot(self) -> go.Figure:
        """Create a box plot of execution times by query type."""
        exec_times = []
        mean_times = []

        for embedder, data in self.results.items():
            if 'execution_times' in data:
                for query_type, times in data['execution_times'].items():
                    for time in times:
                        exec_times.append({
                            'Embedder': embedder,
                            'Query Type': query_type,
                            'Time (s)': time
                        })
            else:
                # Use mean execution time if detailed timing data not available
                mean_times.append({
                    'Embedder': embedder,
                    'Time (s)': data['metrics']['mean_execution_time']
                })

        if exec_times:
            df = pd.DataFrame(exec_times)
            fig = px.box(
                df,
                x='Embedder',
                y='Time (s)',
                color='Query Type',
                title='Query Execution Time Distribution by Query Type'
            )
        else:
            df = pd.DataFrame(mean_times)
            fig = px.bar(
                df,
                x='Embedder',
                y='Time (s)',
                title='Mean Query Execution Time'
            )

        return fig

    def generate_report(self, output_dir: str):
        """Generate a complete HTML report with all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create all visualizations
        metrics_fig = self.create_metrics_comparison()
        query_type_fig = self.create_query_type_performance()
        time_fig = self.create_execution_time_plot()

        # Combine into HTML report
        html_content = f"""
        <html>
        <head>
            <title>RAG Evaluation Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .description {{ color: #666; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>RAG Evaluation Results</h1>

            <div class="chart">
                <h2>Overall Performance Metrics</h2>
                <div class="description">
                    Comparison of semantic and keyword-based retrieval performance
                    across different embedding strategies.
                </div>
                {metrics_fig.to_html(full_html=False)}
            </div>

            <div class="chart">
                <h2>Query Type Performance</h2>
                <div class="description">
                    Performance comparison across different types of queries:
                    semantic-only, keyword-only, and combined approaches.
                </div>
                {query_type_fig.to_html(full_html=False)}
            </div>

            <div class="chart">
                <h2>Execution Time Analysis</h2>
                <div class="description">
                    Query execution time distribution by embedder and query type.
                </div>
                {time_fig.to_html(full_html=False)}
            </div>

            <footer>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """

        with open(output_path / 'evaluation_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Save individual figures
        metrics_fig.write_image(output_path / 'metrics_comparison.png')
        query_type_fig.write_image(output_path / 'query_type_performance.png')
        time_fig.write_image(output_path / 'execution_time.png')
