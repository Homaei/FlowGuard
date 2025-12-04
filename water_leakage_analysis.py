"""
Advanced Water Leakage Detection Analysis
This script analyzes water consumption data to detect potential leaks using multiple methods.
"""

import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class WaterLeakageAnalyzer:
    """
    Comprehensive water leakage detection system using multiple advanced methods.
    """
    
    def __init__(self, csv_file_path=None, dataframe=None):
        """
        Initialize the analyzer with CSV file path or DataFrame.
        
        Parameters:
        csv_file_path (str, optional): Path to the CSV file containing water consumption data
        dataframe (pd.DataFrame, optional): Pre-loaded DataFrame
        """
        self.csv_file_path = csv_file_path
        self.df = dataframe
        self.hourly_data = None
        self.leakage_indicators = {}
        
    def load_data(self):
        """Load and parse the CSV file if DataFrame is not already provided."""
        if self.df is not None:
            print(f"Using provided DataFrame with {len(self.df)} records")
            print(f"Columns: {list(self.df.columns)}")
            return True
            
        if not self.csv_file_path:
            print("Error: No CSV file path provided and no DataFrame loaded.")
            return False

        print(f"Loading data from {self.csv_file_path}...")
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.df)} records")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def parse_volume_registers(self, register_data):
        """
        Parse the message_volume_registers to extract hourly consumption data.
        Handles both string format (from CSV) and list format (from DB/psycopg2).
        
        Parameters:
        register_data: String or List containing hourly volume data
        
        Returns:
        list: List of tuples (timestamp, volume)
        """
        if register_data is None or register_data == '':
            return []
            
        parsed_data = []
        
        # Case 1: It's already a list (from PostgreSQL ARRAY type)
        if isinstance(register_data, list):
            # Format: ['2025-12-03 16:00:00+00:00 => 0.0 m3', ...]
            for item in register_data:
                try:
                    if '=>' in item:
                        parts = item.split('=>')
                        timestamp_str = parts[0].strip()
                        volume_str = parts[1].strip().replace(' m3', '').replace('"', '')
                        
                        # Parse timestamp
                        timestamp = pd.to_datetime(timestamp_str)
                        volume = float(volume_str)
                        parsed_data.append((timestamp, volume))
                except Exception as e:
                    continue
            return parsed_data

        # Case 2: It's a string (from CSV or JSON string)
        if isinstance(register_data, str):
            # Parse the string format: {"timestamp => volume", ...}
            pattern = r'"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\+\-]\d{2}:\d{2}) => ([\d\.]+) m3"'
            matches = re.findall(pattern, register_data)
            
            for timestamp_str, volume_str in matches:
                try:
                    # Parse timestamp (handling timezone)
                    timestamp_str_clean = timestamp_str.replace('+00:00', '').replace('-00:00', '')
                    if '+' in timestamp_str_clean or timestamp_str_clean.count('-') > 2:
                        timestamp = pd.to_datetime(timestamp_str_clean.split('+')[0].split('-')[0] if '+' in timestamp_str_clean else timestamp_str_clean)
                    else:
                        timestamp = pd.to_datetime(timestamp_str_clean)
                    
                    volume = float(volume_str)
                    parsed_data.append((timestamp, volume))
                except Exception as e:
                    continue
        
        return parsed_data
    
    def extract_hourly_data(self):
        """Extract hourly consumption data from all records."""
        print("Extracting hourly consumption data...")
        
        all_hourly_data = []
        
        for idx, row in self.df.iterrows():
            device_name = row.get('device_name', 'unknown')
            message_date = pd.to_datetime(row.get('message_date', ''))
            
            # Parse volume registers
            register_string = row.get('message_volume_registers', '')
            hourly_consumption = self.parse_volume_registers(register_string)
            
            for timestamp, volume in hourly_consumption:
                all_hourly_data.append({
                    'device_name': device_name,
                    'timestamp': timestamp,
                    'volume_m3': volume,
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.dayofweek,
                    'date': timestamp.date()
                })
        
        if all_hourly_data:
            self.hourly_data = pd.DataFrame(all_hourly_data)
            self.hourly_data = self.hourly_data.sort_values('timestamp')
            print(f"Extracted {len(self.hourly_data)} hourly consumption records")
            print(f"Date range: {self.hourly_data['timestamp'].min()} to {self.hourly_data['timestamp'].max()}")
        else:
            print("Warning: No hourly data extracted")
        
        return self.hourly_data is not None
    
    def method1_statistical_analysis(self):
        """
        Method 1: Statistical Analysis - Z-score and Outlier Detection
        Detects consumption values that are significantly different from the mean.
        """
        print("\n=== Method 1: Statistical Analysis (Z-score) ===")
        
        if self.hourly_data is None:
            return
        
        # Calculate statistics
        mean_consumption = self.hourly_data['volume_m3'].mean()
        std_consumption = self.hourly_data['volume_m3'].std()
        median_consumption = self.hourly_data['volume_m3'].median()
        
        # Z-score analysis (threshold: 3 standard deviations)
        z_scores = np.abs(stats.zscore(self.hourly_data['volume_m3']))
        outliers = self.hourly_data[z_scores > 3].copy()
        outliers['z_score'] = z_scores[z_scores > 3]
        
        # IQR method for outlier detection
        Q1 = self.hourly_data['volume_m3'].quantile(0.25)
        Q3 = self.hourly_data['volume_m3'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = self.hourly_data[
            (self.hourly_data['volume_m3'] < lower_bound) | 
            (self.hourly_data['volume_m3'] > upper_bound)
        ].copy()
        
        print(f"Mean consumption: {mean_consumption:.4f} m³")
        print(f"Median consumption: {median_consumption:.4f} m³")
        print(f"Standard deviation: {std_consumption:.4f} m³")
        print(f"Z-score outliers (|z| > 3): {len(outliers)} records")
        print(f"IQR outliers: {len(iqr_outliers)} records")
        
        self.leakage_indicators['statistical'] = {
            'mean': mean_consumption,
            'median': median_consumption,
            'std': std_consumption,
            'z_score_outliers': outliers,
            'iqr_outliers': iqr_outliers,
            'high_consumption_threshold': upper_bound
        }
        
        return outliers, iqr_outliers
    
    def method2_nighttime_consumption_analysis(self):
        """
        Method 2: Nighttime Consumption Analysis
        Detects continuous consumption during nighttime hours (typically 11 PM - 5 AM)
        when no water should be used.
        """
        print("\n=== Method 2: Nighttime Consumption Analysis ===")
        
        if self.hourly_data is None:
            return
        
        # Define nighttime hours (23:00 to 05:00)
        nighttime_hours = [23, 0, 1, 2, 3, 4, 5]
        nighttime_data = self.hourly_data[self.hourly_data['hour'].isin(nighttime_hours)].copy()
        
        # Calculate average nighttime consumption
        nighttime_avg = nighttime_data['volume_m3'].mean()
        
        # Calculate average daytime consumption (06:00 to 22:00)
        daytime_hours = [h for h in range(24) if h not in nighttime_hours]
        daytime_data = self.hourly_data[self.hourly_data['hour'].isin(daytime_hours)].copy()
        daytime_avg = daytime_data['volume_m3'].mean()
        
        # Detect continuous nighttime consumption (potential leak)
        # Threshold: consumption > 0.01 m³ during nighttime
        leak_threshold = 0.01  # 10 liters per hour
        nighttime_leaks = nighttime_data[nighttime_data['volume_m3'] > leak_threshold].copy()
        
        # Group by date to find nights with continuous consumption
        nighttime_by_date = nighttime_data.groupby('date').agg({
            'volume_m3': ['sum', 'mean', 'count'],
            'hour': 'count'
        }).reset_index()
        nighttime_by_date.columns = ['date', 'total_volume', 'avg_volume', 'record_count', 'hour_count']
        
        # Nights with significant consumption
        high_consumption_nights = nighttime_by_date[
            nighttime_by_date['total_volume'] > (leak_threshold * 3)  # More than threshold for 3+ hours
        ].copy()
        
        print(f"Average nighttime consumption: {nighttime_avg:.4f} m³/hour")
        print(f"Average daytime consumption: {daytime_avg:.4f} m³/hour")
        print(f"Nighttime/Daytime ratio: {nighttime_avg/daytime_avg:.2f}" if daytime_avg > 0 else "N/A")
        print(f"Nighttime records above threshold ({leak_threshold} m³): {len(nighttime_leaks)}")
        print(f"Nights with high consumption: {len(high_consumption_nights)}")
        
        self.leakage_indicators['nighttime'] = {
            'nighttime_avg': nighttime_avg,
            'daytime_avg': daytime_avg,
            'nighttime_leaks': nighttime_leaks,
            'high_consumption_nights': high_consumption_nights,
            'ratio': nighttime_avg / daytime_avg if daytime_avg > 0 else 0
        }
        
        return nighttime_leaks, high_consumption_nights
    
    def method3_continuous_flow_detection(self):
        """
        Method 3: Continuous Flow Detection
        Detects periods of continuous, steady consumption that indicate a leak.
        """
        print("\n=== Method 3: Continuous Flow Detection ===")
        
        if self.hourly_data is None:
            return
        
        # Sort by timestamp
        data_sorted = self.hourly_data.sort_values('timestamp').copy()
        
        # Calculate time differences
        data_sorted['time_diff'] = data_sorted['timestamp'].diff()
        data_sorted['volume_diff'] = data_sorted['volume_m3'].diff()
        
        # Detect continuous flow: consistent consumption over multiple hours
        # A leak typically shows: steady consumption > threshold for 3+ consecutive hours
        leak_threshold = 0.005  # 5 liters per hour
        min_consecutive_hours = 3
        
        continuous_flow_periods = []
        current_period = []
        
        for idx, row in data_sorted.iterrows():
            if row['volume_m3'] > leak_threshold:
                if not current_period:
                    current_period = [row]
                else:
                    # Check if this is a continuation (within reasonable time)
                    time_diff = (row['timestamp'] - current_period[-1]['timestamp']).total_seconds() / 3600
                    if time_diff <= 2:  # Within 2 hours
                        current_period.append(row)
                    else:
                        # Period ended, check if it's long enough
                        if len(current_period) >= min_consecutive_hours:
                            continuous_flow_periods.append(current_period)
                        current_period = [row]
            else:
                # Consumption below threshold
                if len(current_period) >= min_consecutive_hours:
                    continuous_flow_periods.append(current_period)
                current_period = []
        
        # Check final period
        if len(current_period) >= min_consecutive_hours:
            continuous_flow_periods.append(current_period)
        
        # Analyze continuous flow periods
        leak_periods_summary = []
        for period in continuous_flow_periods:
            period_df = pd.DataFrame(period)
            leak_periods_summary.append({
                'start_time': period_df['timestamp'].min(),
                'end_time': period_df['timestamp'].max(),
                'duration_hours': len(period),
                'total_volume': period_df['volume_m3'].sum(),
                'avg_volume_per_hour': period_df['volume_m3'].mean(),
                'max_volume': period_df['volume_m3'].max()
            })
        
        leak_periods_df = pd.DataFrame(leak_periods_summary) if leak_periods_summary else pd.DataFrame()
        
        print(f"Continuous flow periods detected: {len(continuous_flow_periods)}")
        if len(leak_periods_df) > 0:
            print(f"Total leakage volume: {leak_periods_df['total_volume'].sum():.4f} m³")
            print(f"Average period duration: {leak_periods_df['duration_hours'].mean():.2f} hours")
        
        self.leakage_indicators['continuous_flow'] = {
            'periods': continuous_flow_periods,
            'summary': leak_periods_df
        }
        
        return leak_periods_df
    
    def method4_isolation_forest(self):
        """
        Method 4: Isolation Forest (Machine Learning)
        Uses unsupervised learning to detect anomalous consumption patterns.
        """
        print("\n=== Method 4: Isolation Forest (Anomaly Detection) ===")
        
        if self.hourly_data is None or len(self.hourly_data) < 10:
            print("Insufficient data for Isolation Forest")
            return
        
        # Prepare features
        features = self.hourly_data[['volume_m3', 'hour', 'day_of_week']].copy()
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply Isolation Forest
        # contamination: expected proportion of outliers (adjust based on domain knowledge)
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        anomaly_labels = iso_forest.fit_predict(features_scaled)
        anomaly_scores = iso_forest.decision_function(features_scaled)
        
        # Get anomalous records
        anomalies = self.hourly_data[anomaly_labels == -1].copy()
        anomalies['anomaly_score'] = anomaly_scores[anomaly_labels == -1]
        
        print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(self.hourly_data)*100:.2f}%)")
        if len(anomalies) > 0:
            print(f"Average anomaly score: {anomalies['anomaly_score'].mean():.4f}")
            print(f"Most anomalous consumption: {anomalies['volume_m3'].max():.4f} m³")
        
        self.leakage_indicators['isolation_forest'] = {
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'model': iso_forest
        }
        
        return anomalies
    
    def method5_baseline_comparison(self):
        """
        Method 5: Baseline Comparison
        Compares current consumption against historical baseline to detect deviations.
        """
        print("\n=== Method 5: Baseline Comparison ===")
        
        if self.hourly_data is None or len(self.hourly_data) < 24:
            print("Insufficient data for baseline comparison")
            return
        
        # Calculate baseline (median consumption for each hour of day)
        baseline = self.hourly_data.groupby('hour')['volume_m3'].agg(['median', 'mean', 'std']).reset_index()
        baseline.columns = ['hour', 'baseline_median', 'baseline_mean', 'baseline_std']
        
        # Merge baseline with hourly data
        data_with_baseline = self.hourly_data.merge(baseline, on='hour', how='left')
        
        # Calculate deviation from baseline
        data_with_baseline['deviation'] = data_with_baseline['volume_m3'] - data_with_baseline['baseline_median']
        data_with_baseline['deviation_percent'] = (data_with_baseline['deviation'] / data_with_baseline['baseline_median'] * 100).replace([np.inf, -np.inf], np.nan)
        
        # Detect significant deviations (more than 2 standard deviations or 50% above baseline)
        significant_deviations = data_with_baseline[
            (data_with_baseline['deviation'] > 2 * data_with_baseline['baseline_std']) |
            (data_with_baseline['deviation_percent'] > 50)
        ].copy()
        
        print(f"Baseline calculated for {len(baseline)} hours")
        print(f"Significant deviations detected: {len(significant_deviations)}")
        if len(significant_deviations) > 0:
            print(f"Maximum deviation: {significant_deviations['deviation'].max():.4f} m³")
            print(f"Average deviation: {significant_deviations['deviation'].mean():.4f} m³")
        
        self.leakage_indicators['baseline'] = {
            'baseline': baseline,
            'deviations': significant_deviations,
            'data_with_baseline': data_with_baseline
        }
        
        return significant_deviations
    
    def method6_pattern_recognition(self):
        """
        Method 6: Pattern Recognition
        Identifies unusual patterns in consumption that may indicate leaks.
        """
        print("\n=== Method 6: Pattern Recognition ===")
        
        if self.hourly_data is None:
            return
        
        # Calculate rolling statistics
        data_sorted = self.hourly_data.sort_values('timestamp').copy()
        data_sorted['rolling_mean_6h'] = data_sorted['volume_m3'].rolling(window=6, min_periods=1).mean()
        data_sorted['rolling_std_6h'] = data_sorted['volume_m3'].rolling(window=6, min_periods=1).std()
        
        # Detect sudden spikes
        spike_threshold = data_sorted['rolling_mean_6h'] + 3 * data_sorted['rolling_std_6h']
        spikes = data_sorted[data_sorted['volume_m3'] > spike_threshold].copy()
        
        # Detect unusual patterns: consumption that's consistently high
        # Group by day and calculate daily statistics
        daily_stats = data_sorted.groupby('date').agg({
            'volume_m3': ['sum', 'mean', 'std', 'max'],
            'timestamp': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'daily_total', 'daily_mean', 'daily_std', 'daily_max', 'hour_count']
        
        # Days with unusually high consumption
        high_consumption_days = daily_stats[
            daily_stats['daily_total'] > daily_stats['daily_total'].quantile(0.75) + 
            1.5 * (daily_stats['daily_total'].quantile(0.75) - daily_stats['daily_total'].quantile(0.25))
        ].copy()
        
        print(f"Consumption spikes detected: {len(spikes)}")
        print(f"Days with high consumption: {len(high_consumption_days)}")
        
        self.leakage_indicators['pattern_recognition'] = {
            'spikes': spikes,
            'high_consumption_days': high_consumption_days,
            'daily_stats': daily_stats
        }
        
        return spikes, high_consumption_days
    
    def method7_dbscan_clustering(self):
        """
        Method 7: DBSCAN Clustering
        Uses density-based clustering to identify unusual consumption patterns.
        """
        print("\n=== Method 7: DBSCAN Clustering ===")
        
        if self.hourly_data is None or len(self.hourly_data) < 20:
            print("Insufficient data for DBSCAN")
            return
        
        # Prepare features: volume, hour, day_of_week
        features = self.hourly_data[['volume_m3', 'hour', 'day_of_week']].values
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN
        # eps: maximum distance between samples in the same neighborhood
        # min_samples: minimum number of samples in a neighborhood
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Identify noise points (label = -1) which could indicate anomalies
        noise_points = self.hourly_data[cluster_labels == -1].copy()
        noise_points['cluster_label'] = cluster_labels[cluster_labels == -1]
        
        # Identify small clusters (potential anomalies)
        unique_labels = set(cluster_labels)
        small_clusters = []
        for label in unique_labels:
            if label != -1:  # Not noise
                cluster_size = sum(cluster_labels == label)
                if cluster_size < len(self.hourly_data) * 0.05:  # Less than 5% of data
                    small_clusters.append(label)
        
        # Filter small cluster points
        if len(small_clusters) > 0:
            small_cluster_mask = pd.Series([label in small_clusters for label in cluster_labels])
            small_cluster_points = self.hourly_data[small_cluster_mask].copy()
        else:
            small_cluster_points = pd.DataFrame()
        
        print(f"Number of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
        print(f"Noise points (anomalies): {len(noise_points)}")
        print(f"Small clusters (potential anomalies): {len(small_cluster_points)}")
        
        self.leakage_indicators['dbscan'] = {
            'noise_points': noise_points,
            'small_clusters': small_cluster_points,
            'cluster_labels': cluster_labels
        }
        
        return noise_points, small_cluster_points
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive leakage detection report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE WATER LEAKAGE DETECTION REPORT")
        print("="*80)
        
        if self.hourly_data is None:
            print("No data available for analysis")
            return
        
        # Overall statistics
        print("\n--- Overall Statistics ---")
        print(f"Total records analyzed: {len(self.hourly_data)}")
        print(f"Date range: {self.hourly_data['timestamp'].min()} to {self.hourly_data['timestamp'].max()}")
        print(f"Total consumption: {self.hourly_data['volume_m3'].sum():.4f} m³")
        print(f"Average hourly consumption: {self.hourly_data['volume_m3'].mean():.4f} m³")
        print(f"Median hourly consumption: {self.hourly_data['volume_m3'].median():.4f} m³")
        print(f"Maximum hourly consumption: {self.hourly_data['volume_m3'].max():.4f} m³")
        
        # Combine all leakage indicators
        all_leak_indicators = []
        
        # Method 1: Statistical
        if 'statistical' in self.leakage_indicators:
            outliers = self.leakage_indicators['statistical']['z_score_outliers']
            if len(outliers) > 0:
                all_leak_indicators.extend(outliers.to_dict('records'))
        
        # Method 2: Nighttime
        if 'nighttime' in self.leakage_indicators:
            nighttime_leaks = self.leakage_indicators['nighttime']['nighttime_leaks']
            if len(nighttime_leaks) > 0:
                all_leak_indicators.extend(nighttime_leaks.to_dict('records'))
        
        # Method 4: Isolation Forest
        if 'isolation_forest' in self.leakage_indicators:
            anomalies = self.leakage_indicators['isolation_forest']['anomalies']
            if len(anomalies) > 0:
                all_leak_indicators.extend(anomalies.to_dict('records'))
        
        # Create unique leak indicators (remove duplicates)
        if all_leak_indicators:
            leak_df = pd.DataFrame(all_leak_indicators)
            leak_df = leak_df.drop_duplicates(subset=['timestamp', 'volume_m3'])
            
            print(f"\n--- Leakage Indicators Summary ---")
            print(f"Total unique leakage indicators: {len(leak_df)}")
            print(f"Percentage of records flagged: {len(leak_df)/len(self.hourly_data)*100:.2f}%")
            
            if len(leak_df) > 0:
                print(f"\nTop 10 highest consumption events:")
                top_consumption = leak_df.nlargest(10, 'volume_m3')[['timestamp', 'volume_m3', 'hour']]
                print(top_consumption.to_string(index=False))
        
        # Risk assessment
        print(f"\n--- Risk Assessment ---")
        risk_score = 0
        
        if 'nighttime' in self.leakage_indicators:
            ratio = self.leakage_indicators['nighttime'].get('ratio', 0)
            if ratio > 0.5:
                risk_score += 3
                print(f"⚠️  HIGH RISK: Nighttime consumption is {ratio:.2f}x daytime consumption")
            elif ratio > 0.3:
                risk_score += 2
                print(f"⚠️  MEDIUM RISK: Nighttime consumption is {ratio:.2f}x daytime consumption")
        
        if 'continuous_flow' in self.leakage_indicators:
            periods = self.leakage_indicators['continuous_flow'].get('summary', pd.DataFrame())
            if len(periods) > 0:
                risk_score += min(len(periods), 5)
                print(f"⚠️  RISK: {len(periods)} continuous flow periods detected")
        
        if 'statistical' in self.leakage_indicators:
            outliers = self.leakage_indicators['statistical'].get('z_score_outliers', pd.DataFrame())
            if len(outliers) > 10:
                risk_score += 2
                print(f"⚠️  RISK: {len(outliers)} statistical outliers detected")
        
        # Overall risk level
        if risk_score >= 7:
            print(f"\n🔴 OVERALL RISK LEVEL: HIGH (Score: {risk_score})")
            print("   Immediate investigation recommended")
        elif risk_score >= 4:
            print(f"\n🟡 OVERALL RISK LEVEL: MEDIUM (Score: {risk_score})")
            print("   Monitoring and investigation recommended")
        elif risk_score >= 1:
            print(f"\n🟢 OVERALL RISK LEVEL: LOW (Score: {risk_score})")
            print("   Continue monitoring")
        else:
            print(f"\n✅ OVERALL RISK LEVEL: MINIMAL (Score: {risk_score})")
            print("   No significant leakage indicators detected")
        
        return leak_df if all_leak_indicators else pd.DataFrame()
    
    def visualize_results(self, output_dir='./'):
        """Generate visualizations of the analysis results."""
        print("\nGenerating visualizations...")
        
        if self.hourly_data is None:
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Time series of consumption
        ax1 = plt.subplot(3, 2, 1)
        self.hourly_data.plot(x='timestamp', y='volume_m3', ax=ax1, alpha=0.6)
        ax1.set_title('Water Consumption Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Volume (m³)')
        ax1.grid(True, alpha=0.3)
        
        # Highlight outliers if available
        if 'statistical' in self.leakage_indicators:
            outliers = self.leakage_indicators['statistical']['z_score_outliers']
            if len(outliers) > 0:
                ax1.scatter(outliers['timestamp'], outliers['volume_m3'], 
                           color='red', s=50, label='Outliers', zorder=5)
                ax1.legend()
        
        # 2. Consumption by hour of day
        ax2 = plt.subplot(3, 2, 2)
        hourly_avg = self.hourly_data.groupby('hour')['volume_m3'].mean()
        hourly_avg.plot(kind='bar', ax=ax2, color='steelblue')
        ax2.set_title('Average Consumption by Hour of Day', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Volume (m³)')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=hourly_avg.quantile(0.75), color='red', linestyle='--', 
                   label='75th percentile', alpha=0.7)
        ax2.legend()
        
        # 3. Distribution of consumption
        ax3 = plt.subplot(3, 2, 3)
        self.hourly_data['volume_m3'].hist(bins=50, ax=ax3, color='steelblue', edgecolor='black')
        ax3.axvline(self.hourly_data['volume_m3'].mean(), color='red', 
                   linestyle='--', label=f'Mean: {self.hourly_data["volume_m3"].mean():.4f}')
        ax3.axvline(self.hourly_data['volume_m3'].median(), color='green', 
                   linestyle='--', label=f'Median: {self.hourly_data["volume_m3"].median():.4f}')
        ax3.set_title('Distribution of Hourly Consumption', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Volume (m³)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Nighttime vs Daytime consumption
        ax4 = plt.subplot(3, 2, 4)
        nighttime_hours = [23, 0, 1, 2, 3, 4, 5]
        self.hourly_data['period'] = self.hourly_data['hour'].apply(
            lambda x: 'Nighttime' if x in nighttime_hours else 'Daytime'
        )
        period_consumption = self.hourly_data.groupby('period')['volume_m3'].mean()
        period_consumption.plot(kind='bar', ax=ax4, color=['darkblue', 'lightblue'])
        ax4.set_title('Nighttime vs Daytime Average Consumption', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Volume (m³)')
        ax4.set_xlabel('Period')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
        
        # 5. Daily consumption trend
        ax5 = plt.subplot(3, 2, 5)
        daily_consumption = self.hourly_data.groupby('date')['volume_m3'].sum()
        daily_consumption.plot(ax=ax5, color='steelblue', marker='o', markersize=4)
        ax5.set_title('Daily Total Consumption', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Total Volume (m³)')
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Box plot by hour
        ax6 = plt.subplot(3, 2, 6)
        hourly_data_pivot = [self.hourly_data[self.hourly_data['hour'] == h]['volume_m3'].values 
                            for h in range(24)]
        bp = ax6.boxplot(hourly_data_pivot, labels=range(24), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax6.set_title('Consumption Distribution by Hour', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Hour of Day')
        ax6.set_ylabel('Volume (m³)')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = f"{output_dir}water_leakage_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to {output_path}")
        plt.close()
        
        return output_path
    
    def run_all_analyses(self):
        """Run all analysis methods."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE WATER LEAKAGE ANALYSIS")
        print("="*80)
        
        # Load data
        if not self.load_data():
            return False
        
        # Extract hourly data
        if not self.extract_hourly_data():
            return False
        
        # Run all analysis methods
        self.method1_statistical_analysis()
        self.method2_nighttime_consumption_analysis()
        self.method3_continuous_flow_detection()
        self.method4_isolation_forest()
        self.method5_baseline_comparison()
        self.method6_pattern_recognition()
        
        # DBSCAN may fail with small datasets, so we'll catch the exception
        try:
            self.method7_dbscan_clustering()
        except Exception as e:
            print(f"DBSCAN analysis skipped: {e}")
        
        # Generate report
        leak_df = self.generate_comprehensive_report()
        
        # Generate visualizations
        # self.visualize_results()
        
        return True


    def get_results_summary(self):
        """
        Get a summary of the analysis results for API response.
        
        Returns:
        dict: Structured analysis results
        """
        # Ensure analysis has been run
        if not self.leakage_indicators:
            self.run_all_analyses()
            
        summary = {
            'is_leaking': False,
            'risk_score': 0,
            'details': {}
        }
        
        risk_score = 0
        
        # Method 1: Statistical
        if 'statistical' in self.leakage_indicators:
            outliers = self.leakage_indicators['statistical'].get('z_score_outliers', pd.DataFrame())
            summary['details']['method1_statistical'] = {
                'outliers_count': len(outliers),
                'status': 'LEAK' if len(outliers) > 5 else 'NORMAL'
            }
            if len(outliers) > 5: risk_score += 2
            
        # Method 2: Nighttime
        if 'nighttime' in self.leakage_indicators:
            ratio = self.leakage_indicators['nighttime'].get('ratio', 0)
            leaks = self.leakage_indicators['nighttime'].get('nighttime_leaks', pd.DataFrame())
            summary['details']['method2_nighttime'] = {
                'ratio': float(f"{ratio:.2f}"),
                'leak_events': len(leaks),
                'status': 'CRITICAL' if ratio > 0.5 else 'WARNING' if ratio > 0.3 else 'NORMAL'
            }
            if ratio > 0.5: risk_score += 3
            elif ratio > 0.3: risk_score += 2
            
        # Method 3: Continuous Flow
        if 'continuous_flow' in self.leakage_indicators:
            periods = self.leakage_indicators['continuous_flow'].get('summary', pd.DataFrame())
            summary['details']['method3_continuous_flow'] = {
                'periods_count': len(periods),
                'total_volume': float(f"{periods['total_volume'].sum():.4f}") if not periods.empty else 0,
                'status': 'LEAK' if len(periods) > 0 else 'NORMAL'
            }
            if len(periods) > 0: risk_score += min(len(periods), 5)

        # Method 4: Isolation Forest
        if 'isolation_forest' in self.leakage_indicators:
            anomalies = self.leakage_indicators['isolation_forest'].get('anomalies', pd.DataFrame())
            summary['details']['method4_isolation_forest'] = {
                'anomalies_count': len(anomalies),
                'status': 'LEAK' if len(anomalies) > 0 else 'NORMAL'
            }
            
        # Method 5: Baseline
        if 'baseline' in self.leakage_indicators:
            deviations = self.leakage_indicators['baseline'].get('deviations', pd.DataFrame())
            summary['details']['method5_baseline'] = {
                'deviations_count': len(deviations),
                'status': 'LEAK' if len(deviations) > 0 else 'NORMAL'
            }

        # Method 6: Pattern Recognition
        if 'pattern_recognition' in self.leakage_indicators:
            spikes = self.leakage_indicators['pattern_recognition'].get('spikes', pd.DataFrame())
            summary['details']['method6_pattern'] = {
                'spikes_count': len(spikes),
                'status': 'LEAK' if len(spikes) > 0 else 'NORMAL'
            }

        # Method 7: DBSCAN
        if 'dbscan' in self.leakage_indicators:
            noise = self.leakage_indicators['dbscan'].get('noise_points', pd.DataFrame())
            summary['details']['method7_dbscan'] = {
                'noise_points': len(noise),
                'status': 'LEAK' if len(noise) > 0 else 'NORMAL'
            }

        summary['risk_score'] = risk_score
        summary['is_leaking'] = risk_score >= 4  # Threshold for overall leak status
        
        return summary

def main():
    """Main function to run the analysis."""
    import sys
    import os
    
    # Get CSV file path from command line or use default
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the smaller file (leakage case)
        csv_file = "data-1762432249318.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        print("Usage: python water_leakage_analysis.py <csv_file_path>")
        return
    
    # Create analyzer instance
    analyzer = WaterLeakageAnalyzer(csv_file)
    
    # Run all analyses
    success = analyzer.run_all_analyses()
    
    if success:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\nKey Files Generated:")
        print("  - water_leakage_analysis.png: Visualizations")
        print("\nNext Steps:")
        print("  1. Review the comprehensive report above")
        print("  2. Examine the visualizations")
        print("  3. Investigate high-risk periods identified")
        print("  4. Consider implementing real-time monitoring")
    else:
        print("Analysis failed. Please check the data file and try again.")


if __name__ == "__main__":
    main()

