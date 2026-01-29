"""
データ分析・可視化モジュール
データの基本統計、型情報、欠損値、外れ値、分布を包括的に分析・可視化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（他で既に設定済みなら尊重）
if not plt.rcParams.get('font.family'):
    plt.rcParams['font.family'] = ['Yu Gothic']
if 'axes.unicode_minus' not in plt.rcParams or plt.rcParams['axes.unicode_minus'] is None:
    plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    """データ分析・可視化クラス"""
    
    def __init__(self, output_folder='./analysis_results'):
        """
        Parameters
        ----------
        output_folder : str
            分析結果の保存先フォルダ
        """
        self.output_folder = output_folder
        import os
        os.makedirs(output_folder, exist_ok=True)
    
    def analyze_dataframe(self, df, target_columns=None, feature_columns=None, 
                          show_plots=True, save_plots=True):
        """
        データフレーム全体の包括的分析
        
        Parameters
        ----------
        df : pd.DataFrame
            分析対象のデータフレーム（生データ）
        target_columns : list
            目的変数のカラム名リスト
        feature_columns : list
            説明変数のカラム名リスト
        show_plots : bool
            グラフを表示するか
        save_plots : bool
            グラフを保存するか
            
        Returns
        -------
        dict
            分析結果の辞書
        """
        print("\n" + "="*80)
        print("データ分析開始")
        print("="*80)
        
        results = {}
        
        # 基本情報
        results['basic_info'] = self._get_basic_info(df)
        self._print_basic_info(results['basic_info'])
        
        # データ型分析
        results['dtype_info'] = self._analyze_dtypes(df)
        self._print_dtype_info(results['dtype_info'])
        
        # 欠損値分析
        results['missing_info'] = self._analyze_missing_values(df)
        self._print_missing_info(results['missing_info'])
        
        # 統計情報
        results['stats_info'] = self._get_statistics(df)
        
        # 外れ値検出（連続変数のみ）
        results['outlier_info'] = self._detect_outliers(df)
        self._print_outlier_info(results['outlier_info'])
        
        # 可視化
        if show_plots or save_plots:
            self._visualize_overview(df, results, show_plots, save_plots)
            
            # 目的変数の詳細分析
            if target_columns:
                for target in target_columns:
                    if target in df.columns:
                        self._analyze_target_variable(df[target], target, show_plots, save_plots)
            
            # 説明変数の詳細分析
            if feature_columns:
                self._analyze_features(df[feature_columns], show_plots, save_plots)
        
        # 相関分析（数値変数のみ）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            results['correlation'] = self._analyze_correlation(df[numeric_cols], show_plots, save_plots)
        
        # レポート保存
        self._save_report(results)
        
        return results
    
    def _get_basic_info(self, df):
        """基本情報の取得"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicated_rows': df.duplicated().sum()
        }
    
    def _print_basic_info(self, info):
        """基本情報の表示"""
        print(f"\n📊 基本情報:")
        print(f"  - データサイズ: {info['shape'][0]} 行 × {info['shape'][1]} 列")
        print(f"  - メモリ使用量: {info['memory_usage']:.2f} MB")
        print(f"  - 重複行数: {info['duplicated_rows']}")
    
    def _analyze_dtypes(self, df):
        """データ型の分析"""
        dtype_counts = df.dtypes.value_counts()
        # dtype型を文字列に変換
        dtype_summary = {str(k): int(v) for k, v in dtype_counts.items()}
        
        dtype_details = {}
        
        for col in df.columns:
            dtype_details[col] = {
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()),
                'unique_ratio': float(df[col].nunique() / len(df)) if len(df) > 0 else 0
            }
        
        return {
            'summary': dtype_summary,
            'details': dtype_details
        }
    
    def _print_dtype_info(self, info):
        """データ型情報の表示"""
        print(f"\n🔤 データ型分布:")
        for dtype, count in info['summary'].items():
            print(f"  - {dtype}: {count} 列")
    
    def _analyze_missing_values(self, df):
        """欠損値の分析"""
        missing_counts = df.isnull().sum()
        missing_ratio = (missing_counts / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_ratio': missing_ratio
        })
        missing_info = missing_info[missing_info['missing_count'] > 0].sort_values('missing_count', ascending=False)
        
        return missing_info
    
    def _print_missing_info(self, info):
        """欠損値情報の表示"""
        if len(info) > 0:
            print(f"\n⚠️ 欠損値情報:")
            for col, row in info.iterrows():
                print(f"  - {col}: {row['missing_count']:.0f} ({row['missing_ratio']:.1f}%)")
        else:
            print(f"\n✅ 欠損値なし")
    
    def _get_statistics(self, df):
        """統計情報の取得"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            stats = numeric_df.describe()
            # 追加統計量
            stats.loc['skewness'] = numeric_df.skew()
            stats.loc['kurtosis'] = numeric_df.kurtosis()
            return stats
        return pd.DataFrame()
    
    def _detect_outliers(self, df, method='iqr', threshold=1.5):
        """
        外れ値の検出（連続変数のみ対象）
        バイナリ変数や離散変数は除外
        """
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            
            # ユニーク値の数を確認
            unique_values = data.nunique()
            
            # 10値以下の変数は外れ値検出をスキップ（バイナリ・離散変数）
            if unique_values <= 10:
                continue
                
            # 連続変数のみ外れ値検出を実行
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
            else:  # zscore
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > threshold]
            
            if len(outliers) > 0:
                outlier_info[col] = {
                    'count': len(outliers),
                    'ratio': len(outliers) / len(data),
                    'values': outliers.tolist() if len(outliers) < 10 else outliers.head(10).tolist(),
                    'method': method,
                    'bounds': (lower_bound, upper_bound) if method == 'iqr' else None
                }
        
        return outlier_info
    
    def _print_outlier_info(self, info):
        """外れ値情報の表示"""
        if info:
            print(f"\n🔍 外れ値検出 (IQR法、連続変数のみ):")
            for col, data in info.items():
                print(f"  - {col}: {data['count']} 個 ({data['ratio']*100:.1f}%)")
                if data['bounds']:
                    print(f"    範囲: [{data['bounds'][0]:.2f}, {data['bounds'][1]:.2f}]")
        else:
            print(f"\n✅ 外れ値なし（連続変数において）")
    
    def _visualize_overview(self, df, results, show=True, save=True):
        """全体的な可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. データ型分布
        ax = axes[0, 0]
        dtype_summary = results['dtype_info']['summary']
        if dtype_summary:
            ax.bar(range(len(dtype_summary)), list(dtype_summary.values()))
            ax.set_xticks(range(len(dtype_summary)))
            ax.set_xticklabels([str(k) for k in dtype_summary.keys()], rotation=45)
            ax.set_title('データ型の分布')
            ax.set_ylabel('列数')
            ax.grid(True, alpha=0.3)
        
        # 2. 欠損値ヒートマップ（上位20列）
        ax = axes[0, 1]
        missing_info = results['missing_info']
        if len(missing_info) > 0:
            top_missing = missing_info.head(20)
            ax.barh(range(len(top_missing)), top_missing['missing_ratio'])
            ax.set_yticks(range(len(top_missing)))
            ax.set_yticklabels(top_missing.index)
            ax.set_xlabel('欠損率 (%)')
            ax.set_title('欠損値の多い変数 (Top 20)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '欠損値なし', ha='center', va='center', fontsize=14)
            ax.set_title('欠損値分析')
        
        # 3. 外れ値サマリー
        ax = axes[1, 0]
        outlier_info = results['outlier_info']
        if outlier_info:
            cols = list(outlier_info.keys())[:10]  # 上位10列
            counts = [outlier_info[c]['count'] for c in cols]
            ax.bar(range(len(cols)), counts)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha='right')
            ax.set_title('外れ値の数 (Top 10、連続変数のみ)')
            ax.set_ylabel('外れ値数')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '外れ値なし', ha='center', va='center', fontsize=14)
            ax.set_title('外れ値分析')
        
        # 4. データ分布の歪度
        ax = axes[1, 1]
        stats_info = results.get('stats_info', pd.DataFrame())
        if 'skewness' in stats_info.index:
            skewness = stats_info.loc['skewness'].sort_values()
            ax.barh(range(len(skewness)), skewness.values)
            ax.set_yticks(range(len(skewness)))
            ax.set_yticklabels(skewness.index, fontsize=8)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('歪度')
            ax.set_title('変数の歪度')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('データ概要分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = f"{self.output_folder}/data_overview.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"\n💾 概要図を保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _analyze_target_variable(self, series, name, show=True, save=True):
        """目的変数の詳細分析"""
        if series.dtype == 'object':
            return  # カテゴリカル変数はスキップ
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. ヒストグラム
        ax = axes[0, 0]
        ax.hist(series.dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'{name}: ヒストグラム')
        ax.set_xlabel('値')
        ax.set_ylabel('頻度')
        ax.grid(True, alpha=0.3)
        
        # 2. ボックスプロット
        ax = axes[0, 1]
        ax.boxplot(series.dropna())
        ax.set_title(f'{name}: ボックスプロット')
        ax.set_ylabel('値')
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Qプロット
        ax = axes[0, 2]
        stats.probplot(series.dropna(), dist="norm", plot=ax)
        ax.set_title(f'{name}: Q-Qプロット')
        ax.grid(True, alpha=0.3)
        
        # 4. 密度プロット
        ax = axes[1, 0]
        series.dropna().plot(kind='density', ax=ax)
        ax.set_title(f'{name}: 確率密度')
        ax.set_xlabel('値')
        ax.grid(True, alpha=0.3)
        
        # 5. 累積分布
        ax = axes[1, 1]
        sorted_data = np.sort(series.dropna())
        ax.plot(sorted_data, np.linspace(0, 1, len(sorted_data)))
        ax.set_title(f'{name}: 累積分布')
        ax.set_xlabel('値')
        ax.set_ylabel('累積確率')
        ax.grid(True, alpha=0.3)
        
        # 6. 統計情報テキスト
        ax = axes[1, 2]
        ax.axis('off')
        
        # 統計値の計算
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        skew_val = series.skew()
        kurt_val = series.kurtosis()
        min_val = series.min()
        max_val = series.max()
        missing_count = series.isnull().sum()
        missing_ratio = missing_count / len(series) * 100
        
        # テキストの作成
        stats_lines = [
            '【統計情報】',
            '',
            f'平均値　　: {mean_val:>10.3f}',
            f'中央値　　: {median_val:>10.3f}',
            f'標準偏差　: {std_val:>10.3f}',
            f'歪度　　　: {skew_val:>10.3f}',
            f'尖度　　　: {kurt_val:>10.3f}',
            f'最小値　　: {min_val:>10.3f}',
            f'最大値　　: {max_val:>10.3f}',
            '',
            f'欠損値　　: {missing_count:>5d} 個',
            f'欠損率　　: {missing_ratio:>10.1f}%'
        ]
        
        stats_text = '\n'.join(stats_lines)
        
        # 表示
        ax.text(0.05, 0.5, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='center', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'目的変数分析: {name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = f"{self.output_folder}/target_{name}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"💾 目的変数分析図を保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _analyze_features(self, df, show=True, save=True):
        """説明変数の分析"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # 数値変数のヒストグラム
            n_cols = min(len(numeric_cols), 20)  # 最大20変数
            n_rows = (n_cols + 3) // 4
            
            fig, axes = plt.subplots(n_rows, 4, figsize=(15, n_rows * 3))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, col in enumerate(numeric_cols[:n_cols]):
                ax = axes[i]
                data = df[col].dropna()
                if data.nunique() <= 1:
                    ax.text(0.5, 0.5, '定数列', ha='center', va='center')
                    ax.set_axis_off()
                    continue
                
                # ヒストグラムと外れ値マーカー
                ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
                
                # IQR法で外れ値検出（連続変数のみ）
                if data.nunique() > 10:  # 連続変数の場合のみ
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    
                    # 外れ値の境界線
                    ax.axvline(lower, color='red', linestyle='--', alpha=0.5, label='外れ値境界')
                    ax.axvline(upper, color='red', linestyle='--', alpha=0.5)
                
                ax.set_title(f'{col}', fontsize=10)
                ax.set_xlabel('値', fontsize=8)
                ax.set_ylabel('頻度', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
            
            # 余分な軸を非表示
            for i in range(n_cols, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle('説明変数の分布（数値変数）', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                save_path = f"{self.output_folder}/features_distribution.png"
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"💾 説明変数分析図を保存: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
    
    def _analyze_correlation(self, df, show=True, save=True):
        """相関分析"""
        # 追加: 安全な数値化（数値以外は NaN にして列ごとに落とす）
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        if df.shape[1] < 2:
            # 相関が計算できない場合は空で返す（描画はスキップ）
            return {'correlation_matrix': pd.DataFrame(), 'high_correlations': []}

        corr_matrix = df.corr()
        
        # 相関の強い変数ペアを抽出
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # 閾値0.7
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if show or save:
            # 相関行列ヒートマップ
            plt.figure(figsize=(12, 10))
            if _HAS_SEABORN:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                            center=0, square=True, linewidths=0.5,
                            cbar_kws={"shrink": 0.8})
            else:
                # フォールバック: matplotlib
                cm = corr_matrix.to_numpy(dtype=float)
                mask = np.triu(np.ones_like(cm, dtype=bool))
                cm_masked = cm.copy()
                cm_masked[mask] = np.nan
                im = plt.imshow(cm_masked, interpolation='nearest', aspect='auto')
                plt.colorbar(im, shrink=0.8)
                ticks = range(len(corr_matrix.columns))
                plt.xticks(ticks, corr_matrix.columns, rotation=90, fontsize=8)
                plt.yticks(ticks, corr_matrix.columns, fontsize=8)
        
            plt.title('相関行列ヒートマップ', fontsize=14, fontweight='bold')
            plt.tight_layout()

            
            if save:
                save_path = f"{self.output_folder}/correlation_heatmap.png"
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"💾 相関行列図を保存: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs
        }
    
    def _save_report(self, results):
        """分析レポートの保存（シンプル版）"""
        import json
        from datetime import datetime
        
        # シンプルなレポート作成
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': {
                'rows': int(results['basic_info']['shape'][0]),
                'columns': int(results['basic_info']['shape'][1])
            },
            'memory_usage_mb': float(results['basic_info']['memory_usage']),
            'duplicated_rows': int(results['basic_info']['duplicated_rows']),
            'missing_columns': len(results['missing_info']),
            'outlier_columns': len(results['outlier_info'])
        }
        
        # 欠損値のトップ10
        if len(results['missing_info']) > 0:
            missing_top10 = {}
            for idx, (col, row) in enumerate(results['missing_info'].head(10).iterrows()):
                missing_top10[str(col)] = {
                    'count': int(row['missing_count']),
                    'ratio': float(row['missing_ratio'])
                }
            report['top_missing'] = missing_top10
        
        # 外れ値情報
        if results['outlier_info']:
            outlier_summary = {}
            for col, info in results['outlier_info'].items():
                outlier_summary[str(col)] = {
                    'count': int(info['count']),
                    'ratio': float(info['ratio'])
                }
            report['outliers'] = outlier_summary
        
        # 高相関ペア
        if 'correlation' in results and results['correlation']:
            if 'high_correlations' in results['correlation']:
                high_corr_list = []
                for pair in results['correlation']['high_correlations']:
                    high_corr_list.append({
                        'var1': str(pair['var1']),
                        'var2': str(pair['var2']),
                        'correlation': float(pair['correlation'])
                    })
                report['high_correlations'] = high_corr_list
        
        # JSON保存
        report_path = f"{self.output_folder}/analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 分析レポートを保存: {report_path}")
        
        # 統計情報をCSVで保存
        if 'stats_info' in results and not results['stats_info'].empty:
            stats_path = f"{self.output_folder}/statistics.csv"
            results['stats_info'].to_csv(stats_path, encoding='utf-8-sig')
            print(f"📊 統計情報を保存: {stats_path}")