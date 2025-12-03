#!/usr/bin/env python3

"""
Read CSV file and convert position & velocity data from camera frame to world frame.
Uses D435 camera intrinsic parameters:
  fx: 525.0              
  fy: 525.0             
  cx: 319.5              
  cy: 239.5              
  depth_factor: 1000

python3 convert_statistics.py input_file.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

class CameraToWorldConverter:
    """Convert tracker statistics from camera coordinate system to world coordinate system."""
    
    def __init__(self):
        # D435 camera intrinsic parameters from config
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5
        self.depth_factor = 1000.0
        
        # Create camera intrinsic matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # Create inverse intrinsic matrix for faster computation
        self.K_inv = np.linalg.inv(self.K)
        
    def camera_to_world(self, x_cam, y_cam, z_cam):
        """
        Convert from camera coordinates to world coordinates.
        
        Args:
            x_cam, y_cam, z_cam: Position in camera coordinate system
            
        Returns:
            x_world, y_world, z_world: Position in world coordinate system
        """
        # In camera frame: x_cam, y_cam are already in meters (from tracker)
        # z_cam is the depth in meters
        
        # Convert to homogeneous pixel coordinates (assuming tracker gives metric coordinates)
        # If tracker gives pixel coordinates, we need to convert back to metric first
        
        # Assuming the tracker data is already in metric coordinates relative to camera
        # Apply coordinate system transformation: camera to world
        # Camera frame: +X right, +Y down, +Z forward
        # World frame: +X right, +Y forward, +Z up (typical ROS convention)
        
        # Transform coordinates from camera frame to world frame
        x_world = x_cam      # X remains the same (right direction)
        y_world = -z_cam     # Camera Z becomes world -Y (forward)
        z_world = -y_cam     # Camera -Y becomes world Z (up)

        ### TEST ONLY ####
        y_world = y_cam  #
        z_world = z_cam  #
        ##################

        return x_world, y_world, z_world
    
    def convert_velocity(self, vx_cam, vy_cam, vz_cam):
        """
        Convert velocity from camera coordinates to world coordinates.
        
        Args:
            vx_cam, vy_cam, vz_cam: Velocity in camera coordinate system
            
        Returns:
            vx_world, vy_world, vz_world: Velocity in world coordinate system
        """
        # Apply same transformation to velocity vectors
        vx_world = vx_cam      # X velocity remains the same
        vy_world = -vz_cam     # Camera Z velocity becomes world -Y velocity
        vz_world = -vy_cam     # Camera -Y velocity becomes world Z velocity
        
        return vx_world, vy_world, vz_world
    
    def process_csv(self, input_file, output_file=None):
        """
        Process the CSV file and convert coordinates.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (optional)
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
            
            # Convert positions from camera frame to world frame
            world_positions = np.array([
                self.camera_to_world(row['pos_x'], row['pos_y'], row['pos_z']) 
                for _, row in df.iterrows()
            ])
            
            # Convert velocities from camera frame to world frame
            world_velocities = np.array([
                self.convert_velocity(row['vel_x'], row['vel_y'], row['vel_z'])
                for _, row in df.iterrows()
            ])
            
            # Create new DataFrame with converted coordinates
            df_converted = df.copy()
            df_converted['pos_x'] = world_positions[:, 0]
            df_converted['pos_y'] = world_positions[:, 1] 
            df_converted['pos_z'] = world_positions[:, 2]
            df_converted['vel_x'] = world_velocities[:, 0]
            df_converted['vel_y'] = world_velocities[:, 1]
            df_converted['vel_z'] = world_velocities[:, 2]
            
            # Generate output filename if not provided
            if output_file is None:
                input_path = Path(input_file)
                output_file = input_path.parent / f"{input_path.stem}_world_coords{input_path.suffix}"
            
            # Save the converted data
            print(f"Saving converted data to: {output_file}")
            df_converted.to_csv(output_file, index=False)
            
            # Print statistics
            print("\nConversion Statistics:")
            print(f"Original position range:")
            print(f"  X: [{df['pos_x'].min():.3f}, {df['pos_x'].max():.3f}]")
            print(f"  Y: [{df['pos_y'].min():.3f}, {df['pos_y'].max():.3f}]") 
            print(f"  Z: [{df['pos_z'].min():.3f}, {df['pos_z'].max():.3f}]")
            
            print(f"Converted position range:")
            print(f"  X: [{df_converted['pos_x'].min():.3f}, {df_converted['pos_x'].max():.3f}]")
            print(f"  Y: [{df_converted['pos_y'].min():.3f}, {df_converted['pos_y'].max():.3f}]")
            print(f"  Z: [{df_converted['pos_z'].min():.3f}, {df_converted['pos_z'].max():.3f}]")
            
            print(f"\nConversion completed successfully!")
            return output_file
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return None

def main():
    """Main function to handle command line arguments and run conversion."""
    parser = argparse.ArgumentParser(description='Convert tracker statistics from camera frame to world frame')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-o', '--output', help='Path to output CSV file (optional)')
    parser.add_argument('--show-params', action='store_true', help='Show camera parameters being used')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    # Create converter instance
    converter = CameraToWorldConverter()
    
    # Show camera parameters if requested
    if args.show_params:
        print("Camera Parameters:")
        print(f"  fx: {converter.fx}")
        print(f"  fy: {converter.fy}")
        print(f"  cx: {converter.cx}")
        print(f"  cy: {converter.cy}")
        print(f"  depth_factor: {converter.depth_factor}")
        print(f"  Intrinsic Matrix K:")
        print(f"    {converter.K}")
        print()
    
    # Process the CSV file
    output_file = converter.process_csv(args.input_file, args.output)
    
    if output_file:
        print(f"\nOutput saved to: {output_file}")
    else:
        print("Conversion failed!")

if __name__ == "__main__":
    main()
