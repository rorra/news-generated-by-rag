"""
Visualization Utilities for RAG Evaluation Results

This module provides functions for visualizing and comparing the performance
of different embedding strategies using various metrics and charts.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


class ResultsVisualizer:
    """
    Class for creating visualizations of RAG evaluation results.

    This class provides methods to generate various charts and comparisons
    of different embedding strategies' performance.
    """

    def __init__(self, results_dir: str):
        """
        Initialize the visualizer with results directory.

        Parameters
        ----------
        results_dir : str
            Path to directory containing evaluation results
        """
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
        """
        Create a bar chart comparing key metrics across embedders.

        Returns
        -------
        go.Figure
            Plotly figure with metrics comparison
        """
        df_metrics = []

        for embedder, data in self.results.items():
            metrics = data['metrics']
            df_metrics.append({
                'Embedder': embedder,
                'Precision@k': metrics['precision_at_k'],
                'Recall@k': metrics['recall_at_k'],
                'NDCG': metrics['ndcg'],
                'QPS': metrics['queries_per_second']
            })

        df = pd.DataFrame(df_metrics)

        # Create subplot with shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Quality Metrics', 'Performance (Queries/Second)'),
            vertical_spacing=0.2
        )

        # Quality metrics
        for metric in ['Precision@k', 'Recall@k', 'NDCG']:
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

        # Performance metric
        fig.add_trace(
            go.Bar(
                name='QPS',
                x=df['Embedder'],
                y=df['QPS'],
                text=df['QPS'].round(2),
                textposition='auto',
            ),
            row=2, col=1
        )

        fig.update_layout(
            title_text='Embedding Strategies Comparison',
            height=800,
            showlegend=True,
            barmode='group'
        )

        return fig

    def create_section_performance(self) -> go.Figure:
        """
        Create a heatmap showing performance across different sections.

        Returns
        -------
        go.Figure
            Plotly figure with section performance heatmap
        """
        section_data = []

        for embedder, data in self.results.items():
            if 'query_categories' in data:
                for section, metrics in data['query_categories'].items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        section_data.append({
                            'Embedder': embedder,
                            'Section': section,
                            'Precision': metrics['precision']
                        })

        if not section_data:
            raise ValueError("No section performance data found in results")

        df = pd.DataFrame(section_data)

        fig = go.Figure(data=go.Heatmap(
            z=pd.pivot_table(
                df,
                values='Precision',
                index='Embedder',
                columns='Section'
            ),
            x=df['Section'].unique(),
            y=df['Embedder'].unique(),
            text=df['Precision'].round(3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdYlGn'
        ))

        fig.update_layout(
            title='Section Performance by Embedder',
            xaxis_title='Section',
            yaxis_title='Embedder'
        )

        return fig

    def create_query_type_comparison(self) -> go.Figure:
        """
        Create a grouped bar chart comparing performance across query types.

        Returns
        -------
        go.Figure
            Plotly figure with query type comparison
        """
        query_data = []

        for embedder, data in self.results.items():
            categories = data.get('query_categories', {})
            query_data.append({
                'Embedder': embedder,
                'Date+Section': categories.get('date_and_section', 0),
                'Date Only': categories.get('date_only', 0),
                'Section Only': categories.get('section_only', 0),
                'No Filters': categories.get('no_filters', 0)
            })

        df = pd.DataFrame(query_data)

        fig = go.Figure()

        for col in ['Date+Section', 'Date Only', 'Section Only', 'No Filters']:
            fig.add_trace(go.Bar(
                name=col,
                x=df['Embedder'],
                y=df[col],
                text=df[col],
                textposition='auto',
            ))

        fig.update_layout(
            title='Query Type Distribution by Embedder',
            xaxis_title='Embedder',
            yaxis_title='Number of Queries',
            barmode='group'
        )

        return fig

    def create_execution_time_plot(self) -> go.Figure:
        """
        Create a box plot of execution times.

        Returns
        -------
        go.Figure
            Plotly figure with execution time distribution
        """
        exec_times = []

        for embedder, data in self.results.items():
            if 'execution_times' in data:
                for time in data['execution_times']:
                    exec_times.append({
                        'Embedder': embedder,
                        'Time (s)': time
                    })

        if exec_times:
            df = pd.DataFrame(exec_times)
            fig = px.box(
                df,
                x='Embedder',
                y='Time (s)',
                title='Query Execution Time Distribution'
            )
        else:
            # Create alternative visualization if no detailed timing data
            mean_times = [{
                'Embedder': emb,
                'Time (s)': data['metrics']['mean_execution_time']
            } for emb, data in self.results.items()]

            df = pd.DataFrame(mean_times)
            fig = px.bar(
                df,
                x='Embedder',
                y='Time (s)',
                title='Mean Query Execution Time'
            )

        return fig

    def generate_report(self, output_dir: str):
        """
        Generate a complete HTML report with all visualizations.

        Parameters
        ----------
        output_dir : str
            Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create all visualizations
        metrics_fig = self.create_metrics_comparison()
        section_fig = self.create_section_performance()
        query_fig = self.create_query_type_comparison()
        time_fig = self.create_execution_time_plot()

        # Combine into HTML report
        html_content = f"""
        <html>
        <head>
            <title>RAG Evaluation Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>RAG Evaluation Results</h1>

            <div class="chart">
                <h2>Overall Metrics Comparison</h2>
                {metrics_fig.to_html(full_html=False)}
            </div>

            <div class="chart">
                <h2>Section Performance</h2>
                {section_fig.to_html(full_html=False)}
            </div>

            <div class="chart">
                <h2>Query Type Distribution</h2>
                {query_fig.to_html(full_html=False)}
            </div>

            <div class="chart">
                <h2>Execution Time Analysis</h2>
                {time_fig.to_html(full_html=False)}
            </div>
        </body>
        </html>
        """

        with open(output_path / 'evaluation_report.html', 'w') as f:
            f.write(html_content)

        # Save individual figures
        metrics_fig.write_image(output_path / 'metrics_comparison.png')
        section_fig.write_image(output_path / 'section_performance.png')
        query_fig.write_image(output_path / 'query_distribution.png')
        time_fig.write_image(output_path / 'execution_time.png')
