
#!/usr/bin/env python3

"""
Read gt_pose from a CSV file and filter data of interest based on a specified tracker ID.
Only filter the same time window as the tracker ID.
tracker_0: 1764074202.769091 -- 1764074214.7333622
we extract the data within this time window from gt_pose.csv, save to gt_pose_tracker_0.csv.

-- Usage:
    单个tracker文件处理: python3 clean_gt.py gt_pose.csv --tracker-file tracker_0.csv

    批量处理多个tracker: python3 clean_gt.py gt_pose.csv --tracker-dir tracker_directory/ -o output_dir/

    手动指定时间窗口: python3 clean_gt.py gt_pose.csv --tracker-file dummy --start-time 1764074200 --end-time 1764074210

"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


class GroundTruthCleaner:
    """Filter ground truth data based on tracker time windows."""
    
    def __init__(self):
        pass
    
    def get_tracker_time_window(self, tracker_file):
        """
        Get the time window (start, end) from a tracker CSV file.
        
        Args:
            tracker_file: Path to tracker CSV file
            
        Returns:
            tuple: (start_time, end_time) or None if error
        """
        try:
            df = pd.read_csv(tracker_file)
            
            if 'time' not in df.columns:
                print(f"Error: 'time' column not found in {tracker_file}")
                return None
            
            start_time = df['time'].min()
            end_time = df['time'].max()
            
            return start_time, end_time
            
        except Exception as e:
            print(f"Error reading tracker file {tracker_file}: {e}")
            return None
    
    def filter_gt_by_time_window(self, gt_file, start_time, end_time, output_file=None):
        """
        Filter ground truth data within specified time window.
        
        Args:
            gt_file: Path to ground truth CSV file
            start_time: Start timestamp
            end_time: End timestamp
            output_file: Output file path (optional)
            
        Returns:
            Path to output file or None if error
        """
        try:
            # Read ground truth file
            print(f"Reading ground truth file: {gt_file}")
            df_gt = pd.read_csv(gt_file)
            
            if 'time' not in df_gt.columns:
                print(f"Error: 'time' column not found in {gt_file}")
                return None
            
            print(f"Ground truth data: {len(df_gt)} records")
            print(f"Time range: [{df_gt['time'].min():.6f}, {df_gt['time'].max():.6f}]")
            
            # Filter by time window
            print(f"Filtering time window: [{start_time:.6f}, {end_time:.6f}]")
            mask = (df_gt['time'] >= start_time) & (df_gt['time'] <= end_time)
            df_filtered = df_gt[mask].copy()
            
            print(f"Filtered data: {len(df_filtered)} records")
            
            if len(df_filtered) == 0:
                print("Warning: No data found within specified time window!")
                return None
            
            # Sort by time
            df_filtered = df_filtered.sort_values('time')
            
            # Generate output filename if not provided
            if output_file is None:
                gt_path = Path(gt_file)
                output_file = gt_path.parent / f"{gt_path.stem}_filtered_{start_time:.0f}_{end_time:.0f}{gt_path.suffix}"
            
            # Save filtered data
            print(f"Saving filtered ground truth to: {output_file}")
            df_filtered.to_csv(output_file, index=False)
            
            # Print statistics
            duration = end_time - start_time
            print(f"\nFiltering Statistics:")
            print(f"  Time window duration: {duration:.3f}s")
            print(f"  Records in window: {len(df_filtered)}")
            print(f"  Sampling rate: {len(df_filtered) / duration:.3f} Hz")
            
            return output_file
            
        except Exception as e:
            print(f"Error filtering ground truth data: {e}")
            return None
    
    def process_tracker_and_gt(self, tracker_file, gt_file, output_dir=None):
        """
        Process a tracker file and corresponding ground truth data.
        
        Args:
            tracker_file: Path to tracker CSV file
            gt_file: Path to ground truth CSV file
            output_dir: Output directory (optional)
            
        Returns:
            Path to filtered GT file or None if error
        """
        try:
            # Extract tracker ID from filename
            tracker_path = Path(tracker_file)
            tracker_name = tracker_path.stem
            
            print(f"Processing tracker: {tracker_name}")
            
            # Get time window from tracker file
            time_window = self.get_tracker_time_window(tracker_file)
            if time_window is None:
                return None
            
            start_time, end_time = time_window
            print(f"Tracker time window: [{start_time:.6f}, {end_time:.6f}]")
            print(f"Duration: {end_time - start_time:.3f}s")
            
            # Setup output file
            gt_path = Path(gt_file)
            if output_dir is None:
                output_dir = gt_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"{tracker_name}_GT.csv"
            
            # Filter ground truth data
            result = self.filter_gt_by_time_window(gt_file, start_time, end_time, output_file)
            
            return result
            
        except Exception as e:
            print(f"Error processing tracker and GT: {e}")
            return None
    
    def batch_process(self, tracker_dir, gt_file, output_dir=None):
        """
        Batch process multiple tracker files with the same GT file.
        
        Args:
            tracker_dir: Directory containing tracker CSV files
            gt_file: Path to ground truth CSV file
            output_dir: Output directory (optional)
            
        Returns:
            List of processed output files
        """
        try:
            tracker_dir = Path(tracker_dir)
            
            # Find all tracker CSV files
            tracker_files = list(tracker_dir.glob("tracker_*.csv"))
            
            if not tracker_files:
                print(f"No tracker files found in {tracker_dir}")
                return []
            
            print(f"Found {len(tracker_files)} tracker files")
            
            output_files = []
            
            for tracker_file in sorted(tracker_files):
                print(f"\n{'='*50}")
                result = self.process_tracker_and_gt(tracker_file, gt_file, output_dir)
                if result:
                    output_files.append(result)
            
            print(f"\n{'='*50}")
            print(f"Batch processing completed!")
            print(f"Processed {len(output_files)} out of {len(tracker_files)} tracker files")
            
            return output_files
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return []


def main():
    """Main function to handle command line arguments and run GT cleaning."""
    parser = argparse.ArgumentParser(description='Filter ground truth data based on tracker time windows')
    parser.add_argument('gt_file', help='Path to ground truth CSV file')
    
    # Either specify a single tracker file or a directory of tracker files
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tracker-file', help='Path to single tracker CSV file')
    group.add_argument('--tracker-dir', help='Directory containing tracker CSV files')
    
    parser.add_argument('-o', '--output-dir', help='Output directory (optional)')
    parser.add_argument('--start-time', type=float, help='Manual start time (optional)')
    parser.add_argument('--end-time', type=float, help='Manual end time (optional)')
    
    args = parser.parse_args()
    
    # Check if GT file exists
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file '{args.gt_file}' does not exist.")
        return
    
    # Create cleaner instance
    cleaner = GroundTruthCleaner()
    
    # Process based on input type
    if args.tracker_file:
        # Single tracker file processing
        if args.start_time is not None and args.end_time is not None:
            # Manual time window - don't need to check tracker file
            gt_path = Path(args.gt_file)
            output_dir = Path(args.output_dir) if args.output_dir else gt_path.parent
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"gt_pose_manual_{args.start_time:.0f}_{args.end_time:.0f}.csv"
            result = cleaner.filter_gt_by_time_window(args.gt_file, args.start_time, args.end_time, output_file)
        else:
            # Use tracker file time window
            if not os.path.exists(args.tracker_file):
                print(f"Error: Tracker file '{args.tracker_file}' does not exist.")
                return
            
            result = cleaner.process_tracker_and_gt(args.tracker_file, args.gt_file, args.output_dir)
        
        if result:
            print(f"\nFiltering completed! Output saved to: {result}")
        else:
            print("Filtering failed!")
    
    elif args.tracker_dir:
        # Batch processing
        if not os.path.exists(args.tracker_dir):
            print(f"Error: Tracker directory '{args.tracker_dir}' does not exist.")
            return
        
        output_files = cleaner.batch_process(args.tracker_dir, args.gt_file, args.output_dir)
        
        if output_files:
            print(f"\nBatch processing completed! Generated {len(output_files)} files:")
            for file in output_files:
                print(f"  {file}")
        else:
            print("Batch processing failed!")


if __name__ == "__main__":
    main()