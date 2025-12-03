#!/usr/bin/env python3

"""
Convert origin point of tracker statistics to ground truth origin
Then calculate position error.

This script:
1. Reads ground truth and tracker data
2. Calculates offset between starting points
3. Applies offset to all tracker measurements
4. Computes position error between corrected tracker and ground truth trajectories

# single process
python3 post_process.py gt_file.csv tracker_file.csv -o output_dir/

# batch process
for i in {0..4}; do
  python3 post_process.py gt_filtered/gt_pose_tracker_${i}.csv \
                         tracker_filtered/tracker_${i}.csv \
                         -o analysis_results/
done

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d


class TrajectoryAnalyzer:
    """Analyze tracker trajectories against ground truth with offset correction."""
    
    def __init__(self, xy_only=False):
        self.xy_only = xy_only
    
    def load_data(self, gt_file, tracker_file):
        """
        Load ground truth and tracker data.
        
        Args:
            gt_file: Path to ground truth CSV file
            tracker_file: Path to tracker CSV file
            
        Returns:
            tuple: (gt_data, tracker_data) DataFrames
        """
        try:
            print(f"Loading ground truth: {gt_file}")
            gt_data = pd.read_csv(gt_file)
            
            print(f"Loading tracker data: {tracker_file}")
            tracker_data = pd.read_csv(tracker_file)
            
            # Validate required columns
            gt_required = ['time', 'pos_x', 'pos_y', 'pos_z']
            tracker_required = ['time', 'pos_x', 'pos_y', 'pos_z']
            
            for col in gt_required:
                if col not in gt_data.columns:
                    raise ValueError(f"Missing column '{col}' in ground truth data")
            
            for col in tracker_required:
                if col not in tracker_data.columns:
                    raise ValueError(f"Missing column '{col}' in tracker data")
            
            # Sort by time
            gt_data = gt_data.sort_values('time').reset_index(drop=True)
            tracker_data = tracker_data.sort_values('time').reset_index(drop=True)
            
            print(f"Ground truth: {len(gt_data)} records, time range: [{gt_data['time'].min():.3f}, {gt_data['time'].max():.3f}]")
            print(f"Tracker data: {len(tracker_data)} records, time range: [{tracker_data['time'].min():.3f}, {tracker_data['time'].max():.3f}]")
            
            return gt_data, tracker_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def calculate_initial_offset(self, gt_data, tracker_data):
        """
        Calculate offset between starting points of ground truth and tracker.
        
        Args:
            gt_data: Ground truth DataFrame
            tracker_data: Tracker DataFrame
            
        Returns:
            tuple: (offset_x, offset_y, offset_z)
        """
        try:
            # Get starting positions
            gt_start = np.array([gt_data.iloc[0]['pos_x'], gt_data.iloc[0]['pos_y'], gt_data.iloc[0]['pos_z']])
            tracker_start = np.array([tracker_data.iloc[0]['pos_x'], tracker_data.iloc[0]['pos_y'], tracker_data.iloc[0]['pos_z']])
            
            # Calculate offset (GT - Tracker)
            offset = gt_start - tracker_start
            
            print(f"Starting positions:")
            print(f"  Ground truth: [{gt_start[0]:.3f}, {gt_start[1]:.3f}, {gt_start[2]:.3f}]")
            print(f"  Tracker:      [{tracker_start[0]:.3f}, {tracker_start[1]:.3f}, {tracker_start[2]:.3f}]")
            print(f"  Offset:       [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]")
            
            return offset[0], offset[1], offset[2]
            
        except Exception as e:
            print(f"Error calculating offset: {e}")
            return 0.0, 0.0, 0.0
    
    def apply_offset_correction(self, tracker_data, offset_x, offset_y, offset_z):
        """
        Apply offset correction to tracker data.
        
        Args:
            tracker_data: Tracker DataFrame
            offset_x, offset_y, offset_z: Offset values
            
        Returns:
            DataFrame: Corrected tracker data
        """
        try:
            corrected_data = tracker_data.copy()
            
            # Apply offset correction
            corrected_data['pos_x'] += offset_x
            corrected_data['pos_y'] += offset_y
            corrected_data['pos_z'] += offset_z
            
            print(f"Applied offset correction:")
            print(f"  Original first position: [{tracker_data.iloc[0]['pos_x']:.3f}, {tracker_data.iloc[0]['pos_y']:.3f}, {tracker_data.iloc[0]['pos_z']:.3f}]")
            print(f"  Corrected first position: [{corrected_data.iloc[0]['pos_x']:.3f}, {corrected_data.iloc[0]['pos_y']:.3f}, {corrected_data.iloc[0]['pos_z']:.3f}]")
            
            return corrected_data
            
        except Exception as e:
            print(f"Error applying offset correction: {e}")
            return tracker_data
    
    def interpolate_gt_to_tracker_times(self, gt_data, tracker_data):
        """
        Interpolate ground truth data to match tracker timestamps.
        
        Args:
            gt_data: Ground truth DataFrame
            tracker_data: Tracker DataFrame
            
        Returns:
            DataFrame: Interpolated ground truth data
        """
        try:
            # Find overlapping time range
            gt_time_min, gt_time_max = gt_data['time'].min(), gt_data['time'].max()
            tracker_time_min, tracker_time_max = tracker_data['time'].min(), tracker_data['time'].max()
            
            overlap_start = max(gt_time_min, tracker_time_min)
            overlap_end = min(gt_time_max, tracker_time_max)
            
            # Filter tracker data to overlap range
            tracker_filtered = tracker_data[
                (tracker_data['time'] >= overlap_start) & 
                (tracker_data['time'] <= overlap_end)
            ].copy()
            
            if len(tracker_filtered) == 0:
                print("Warning: No overlapping time range found!")
                return gt_data.iloc[:0].copy(), tracker_filtered
            
            print(f"Overlap time range: [{overlap_start:.3f}, {overlap_end:.3f}]")
            print(f"Tracker points in overlap: {len(tracker_filtered)}")
            
            # Create interpolation functions for GT data
            interp_x = interp1d(gt_data['time'], gt_data['pos_x'], kind='linear', fill_value='extrapolate')
            interp_y = interp1d(gt_data['time'], gt_data['pos_y'], kind='linear', fill_value='extrapolate')
            interp_z = interp1d(gt_data['time'], gt_data['pos_z'], kind='linear', fill_value='extrapolate')
            
            # Interpolate GT to tracker timestamps
            gt_interpolated = pd.DataFrame({
                'time': tracker_filtered['time'],
                'pos_x': interp_x(tracker_filtered['time']),
                'pos_y': interp_y(tracker_filtered['time']),
                'pos_z': interp_z(tracker_filtered['time'])
            })
            
            return gt_interpolated, tracker_filtered
            
        except Exception as e:
            print(f"Error interpolating data: {e}")
            return gt_data.iloc[:0].copy(), tracker_data.iloc[:0].copy()
    
    def calculate_position_errors(self, gt_data, tracker_data):
        """
        Calculate position errors between ground truth and tracker data.
        
        Args:
            gt_data: Ground truth DataFrame (interpolated)
            tracker_data: Corrected tracker DataFrame
            
        Returns:
            DataFrame: Error analysis results
        """
        try:
            if len(gt_data) != len(tracker_data):
                print(f"Warning: Data length mismatch - GT: {len(gt_data)}, Tracker: {len(tracker_data)}")
                min_len = min(len(gt_data), len(tracker_data))
                gt_data = gt_data.iloc[:min_len]
                tracker_data = tracker_data.iloc[:min_len]
            
            # Calculate position differences
            dx = gt_data['pos_x'].values - tracker_data['pos_x'].values
            dy = gt_data['pos_y'].values - tracker_data['pos_y'].values
            dz = gt_data['pos_z'].values - tracker_data['pos_z'].values
            
            # Calculate Euclidean distances
            if self.xy_only:
                euclidean_errors = np.sqrt(dx**2 + dy**2)  # Only XY plane
                print(f"Calculating errors in XY plane only (Z ignored)")
            else:
                euclidean_errors = np.sqrt(dx**2 + dy**2 + dz**2)  # Full 3D
            
            # Create results DataFrame
            errors_df = pd.DataFrame({
                'time': tracker_data['time'].values,
                'gt_x': gt_data['pos_x'].values,
                'gt_y': gt_data['pos_y'].values,
                'gt_z': gt_data['pos_z'].values,
                'tracker_x': tracker_data['pos_x'].values,
                'tracker_y': tracker_data['pos_y'].values,
                'tracker_z': tracker_data['pos_z'].values,
                'error_x': dx,
                'error_y': dy,
                'error_z': dz,
                'euclidean_error': euclidean_errors
            })
            
            # Calculate statistics
            stats = {
                'mean_error': np.mean(euclidean_errors),
                'median_error': np.median(euclidean_errors),
                'std_error': np.std(euclidean_errors),
                'max_error': np.max(euclidean_errors),
                'min_error': np.min(euclidean_errors),
                'rmse': np.sqrt(np.mean(euclidean_errors**2)),
                'mean_error_x': np.mean(np.abs(dx)),
                'mean_error_y': np.mean(np.abs(dy)),
                'mean_error_z': np.mean(np.abs(dz)),
                'std_error_x': np.std(dx),
                'std_error_y': np.std(dy),
                'std_error_z': np.std(dz)
            }
            
            print(f"\nPosition Error Statistics:")
            if self.xy_only:
                print(f"  Mean XY Euclidean Error: {stats['mean_error']:.4f} m")
                print(f"  Median XY Euclidean Error: {stats['median_error']:.4f} m")
                print(f"  XY RMSE: {stats['rmse']:.4f} m")
                print(f"  Max XY Error: {stats['max_error']:.4f} m")
                print(f"  Min XY Error: {stats['min_error']:.4f} m")
                print(f"  Std XY Error: {stats['std_error']:.4f} m")
            else:
                print(f"  Mean Euclidean Error: {stats['mean_error']:.4f} m")
                print(f"  Median Euclidean Error: {stats['median_error']:.4f} m")
                print(f"  RMSE: {stats['rmse']:.4f} m")
                print(f"  Max Error: {stats['max_error']:.4f} m")
                print(f"  Min Error: {stats['min_error']:.4f} m")
                print(f"  Std Error: {stats['std_error']:.4f} m")
            print(f"  Mean Absolute Error - X: {stats['mean_error_x']:.4f} m")
            print(f"  Mean Absolute Error - Y: {stats['mean_error_y']:.4f} m")
            print(f"  Mean Absolute Error - Z: {stats['mean_error_z']:.4f} m")
            
            return errors_df, stats
            
        except Exception as e:
            print(f"Error calculating position errors: {e}")
            return None, None
    
    def save_results(self, errors_df, stats, output_file):
        """
        Save error analysis results to file.
        
        Args:
            errors_df: DataFrame with error data
            stats: Dictionary with statistics
            output_file: Output file path
        """
        try:
            # Save detailed error data
            errors_df.to_csv(output_file, index=False)
            
            # Save statistics to separate file
            output_path = Path(output_file)
            stats_file = output_path.parent / f"{output_path.stem}_stats.txt"
            
            with open(stats_file, 'w') as f:
                f.write("Position Error Analysis Statistics\n")
                f.write("="*50 + "\n\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value:.6f}\n")
            
            print(f"Results saved to: {output_file}")
            print(f"Statistics saved to: {stats_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_error_plots(self, errors_df, output_dir):
        """
        Create visualization plots for error analysis.
        
        Args:
            errors_df: DataFrame with error data
            output_dir: Directory to save plots
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Plot 1: Euclidean error over time
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(errors_df['time'] - errors_df['time'].iloc[0], errors_df['euclidean_error'])
            plt.xlabel('Time (s)')
            if self.xy_only:
                plt.ylabel('XY Euclidean Error (m)')
                plt.title('XY Position Error Over Time')
            else:
                plt.ylabel('Euclidean Error (m)')
                plt.title('Position Error Over Time')
            plt.grid(True)
            
            # Plot 2: XYZ errors over time
            plt.subplot(2, 2, 2)
            time_rel = errors_df['time'] - errors_df['time'].iloc[0]
            plt.plot(time_rel, errors_df['error_x'], label='X Error', alpha=0.7)
            plt.plot(time_rel, errors_df['error_y'], label='Y Error', alpha=0.7)
            plt.plot(time_rel, errors_df['error_z'], label='Z Error', alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('Position Error (m)')
            plt.title('XYZ Position Errors')
            plt.legend()
            plt.grid(True)
            
            # Plot 3: Error histogram
            plt.subplot(2, 2, 3)
            plt.hist(errors_df['euclidean_error'], bins=30, alpha=0.7)
            if self.xy_only:
                plt.xlabel('XY Euclidean Error (m)')
                plt.title('XY Error Distribution')
            else:
                plt.xlabel('Euclidean Error (m)')
                plt.title('Error Distribution')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Plot 4: Trajectory comparison (2D)
            plt.subplot(2, 2, 4)
            plt.plot(errors_df['gt_x'], errors_df['gt_y'], 'b-', label='Ground Truth', linewidth=2)
            plt.plot(errors_df['tracker_x'], errors_df['tracker_y'], 'r--', label='Tracker', linewidth=2, alpha=0.7)
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Trajectory Comparison (XY)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            plt.tight_layout()
            plot_file = output_dir / 'position_error_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Error plots saved to: {plot_file}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def process(self, gt_file, tracker_file, output_dir=None):
        """
        Main processing function.
        
        Args:
            gt_file: Path to ground truth CSV file
            tracker_file: Path to tracker CSV file
            output_dir: Output directory (optional)
            
        Returns:
            tuple: (errors_df, stats) or (None, None) if error
        """
        try:
            # Load data
            gt_data, tracker_data = self.load_data(gt_file, tracker_file)
            if gt_data is None or tracker_data is None:
                return None, None
            
            # Calculate and apply offset correction
            offset_x, offset_y, offset_z = self.calculate_initial_offset(gt_data, tracker_data)
            corrected_tracker = self.apply_offset_correction(tracker_data, offset_x, offset_y, offset_z)
            
            # Interpolate GT to tracker timestamps
            gt_interp, tracker_sync = self.interpolate_gt_to_tracker_times(gt_data, corrected_tracker)
            
            # Calculate position errors
            errors_df, stats = self.calculate_position_errors(gt_interp, tracker_sync)
            if errors_df is None:
                return None, None
            
            # Setup output directory
            if output_dir is None:
                tracker_path = Path(tracker_file)
                output_dir = tracker_path.parent / "error_analysis"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(exist_ok=True)
            
            # Generate output filename
            tracker_path = Path(tracker_file)
            tracker_name = tracker_path.stem
            suffix = "_xy" if self.xy_only else ""
            output_file = output_dir / f"position_errors_{tracker_name}{suffix}.csv"
            
            # Save results
            self.save_results(errors_df, stats, output_file)
            
            # Create plots
            self.create_error_plots(errors_df, output_dir)
            
            return errors_df, stats
            
        except Exception as e:
            print(f"Error in processing: {e}")
            return None, None


def main():
    """Main function to handle command line arguments and run trajectory analysis."""
    parser = argparse.ArgumentParser(description='Analyze tracker trajectory against ground truth')
    parser.add_argument('gt_file', help='Path to ground truth CSV file')
    parser.add_argument('tracker_file', help='Path to tracker CSV file')
    parser.add_argument('-o', '--output-dir', help='Output directory (optional)')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--xy-only', action='store_true', help='Calculate errors in XY plane only (ignore Z)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file '{args.gt_file}' does not exist.")
        return
    
    if not os.path.exists(args.tracker_file):
        print(f"Error: Tracker file '{args.tracker_file}' does not exist.")
        return
    
    # Create analyzer instance
    analyzer = TrajectoryAnalyzer(xy_only=args.xy_only)
    
    # Process the data
    print("Starting trajectory analysis...")
    if args.xy_only:
        print("XY-only mode: Z-axis errors will be ignored")
    
    errors_df, stats = analyzer.process(args.gt_file, args.tracker_file, args.output_dir)
    
    if errors_df is not None and stats is not None:
        print("\nTrajectory analysis completed successfully!")
    else:
        print("Trajectory analysis failed!")


if __name__ == "__main__":
    main()
