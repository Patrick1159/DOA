#!/usr/bin/env python3

"""
Read statistics CSV file and filter out different trackers based on tracker_id.
Each tracker_id's data will be saved as a separate CSV file.

# 基本分类过滤
python3 index_filter.py input_file.csv

# 指定输出目录
python3 index_filter.py input_file.csv -o output_directory

# 显示摘要统计
python3 index_filter.py input_file.csv --summary

# 列出所有tracker ID
python3 index_filter.py input_file.csv --list-trackers
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


class TrackerIndexFilter:
    """Filter tracker statistics by tracker_id and save to separate files."""
    
    def __init__(self):
        pass
    
    def process_csv(self, input_file, output_dir=None):
        """
        Process the CSV file and split by tracker_id.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory to save output files (optional)
        """
        try:
            # Read the CSV file
            print(f"Reading CSV file: {input_file}")
            df = pd.read_csv(input_file)
            
            # Validate required columns
            required_columns = ['time', 'tracker_id', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Processing {len(df)} records...")
            
            # Get unique tracker IDs
            unique_tracker_ids = sorted(df['tracker_id'].unique())
            print(f"Found {len(unique_tracker_ids)} unique tracker IDs: {unique_tracker_ids}")
            
            # Setup output directory
            input_path = Path(input_file)
            if output_dir is None:
                output_dir = input_path.parent / f"{input_path.stem}_filtered"
            else:
                output_dir = Path(output_dir)
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(exist_ok=True)
            print(f"Output directory: {output_dir}")
            
            output_files = []
            
            # Process each tracker ID
            for tracker_id in unique_tracker_ids:
                # Filter data for this tracker
                tracker_data = df[df['tracker_id'] == tracker_id].copy()
                
                # Sort by time
                tracker_data = tracker_data.sort_values('time')
                
                # Generate output filename
                output_file = output_dir / f"tracker_{tracker_id}_{input_path.stem}.csv"
                
                # Save filtered data
                tracker_data.to_csv(output_file, index=False)
                output_files.append(output_file)
                
                # Print statistics for this tracker
                duration = tracker_data['time'].max() - tracker_data['time'].min()
                print(f"  Tracker {tracker_id}: {len(tracker_data)} records, duration: {duration:.3f}s")
                print(f"    Time range: [{tracker_data['time'].min():.3f}, {tracker_data['time'].max():.3f}]")
                print(f"    Position range:")
                print(f"      X: [{tracker_data['pos_x'].min():.3f}, {tracker_data['pos_x'].max():.3f}]")
                print(f"      Y: [{tracker_data['pos_y'].min():.3f}, {tracker_data['pos_y'].max():.3f}]") 
                print(f"      Z: [{tracker_data['pos_z'].min():.3f}, {tracker_data['pos_z'].max():.3f}]")
                print(f"    Saved to: {output_file}")
                print()
            
            print(f"Filtering completed successfully!")
            print(f"Generated {len(output_files)} files in {output_dir}")
            
            return output_files
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return None
    
    def get_summary_statistics(self, input_file):
        """
        Get summary statistics about tracker data.
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            df = pd.read_csv(input_file)
            
            summary = {
                'total_records': len(df),
                'unique_trackers': df['tracker_id'].nunique(),
                'tracker_ids': sorted(df['tracker_id'].unique().tolist()),
                'time_range': [df['time'].min(), df['time'].max()],
                'duration': df['time'].max() - df['time'].min(),
                'records_per_tracker': df['tracker_id'].value_counts().to_dict()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting summary statistics: {e}")
            return None


def main():
    """Main function to handle command line arguments and run filtering."""
    parser = argparse.ArgumentParser(description='Filter tracker statistics by tracker_id')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-o', '--output-dir', help='Directory to save output files (optional)')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics only')
    parser.add_argument('--list-trackers', action='store_true', help='List all tracker IDs in the file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    # Create filter instance
    filter_tool = TrackerIndexFilter()
    
    # Show summary statistics if requested
    if args.summary or args.list_trackers:
        summary = filter_tool.get_summary_statistics(args.input_file)
        if summary:
            print("Summary Statistics:")
            print(f"  Total records: {summary['total_records']}")
            print(f"  Unique trackers: {summary['unique_trackers']}")
            print(f"  Tracker IDs: {summary['tracker_ids']}")
            print(f"  Time range: [{summary['time_range'][0]:.3f}, {summary['time_range'][1]:.3f}]")
            print(f"  Duration: {summary['duration']:.3f}s")
            print(f"  Records per tracker:")
            for tracker_id, count in summary['records_per_tracker'].items():
                print(f"    Tracker {tracker_id}: {count} records")
        
        if args.summary:
            return
    
    # Process the CSV file
    output_files = filter_tool.process_csv(args.input_file, args.output_dir)
    
    if output_files:
        print(f"\nFiltering completed! Generated {len(output_files)} files.")
    else:
        print("Filtering failed!")


if __name__ == "__main__":
    main()
