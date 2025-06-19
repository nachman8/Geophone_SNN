import json
import glob
import os
from pathlib import Path

def read_and_write_json_files(data_directory=".", json_pattern="f_*.json", output_file="resonator_parameters_simple.txt"):
    """
    Read all JSON files and write specific parameters to a text file
    
    Parameters:
    - data_directory: Directory path containing JSON files (e.g., "/home/data/lf")
    - json_pattern: Pattern to match JSON files (default: "f_*.json")
    - output_file: Output text file name
    """
    
    # Construct full path pattern
    full_pattern = os.path.join(data_directory, json_pattern)
    
    # Find all JSON files
    json_files = glob.glob(full_pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {full_pattern}")
        return
    
    # Sort files for consistent output
    json_files.sort()
    
    with open(output_file, 'w') as out_file:
        out_file.write(f"Found {len(json_files)} JSON files\n")
        out_file.write("=" * 80 + "\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                filename = Path(json_file).name
                
                # Extract the required fields
                freq0 = data.get('freq0', 'N/A')
                clk_freq = data.get('clk_freq', 'N/A')
                lf = data.get('lf', 'N/A')
                weight_results = data.get('weight_results', 'N/A')
                theta_results = data.get('theta_results', 'N/A')
                
                out_file.write(f"File: {filename}\n")
                out_file.write(f"  freq0: {freq0}\n")
                out_file.write(f"  clk_freq: {clk_freq}\n")
                out_file.write(f"  lf: {lf}\n")
                out_file.write(f"  weight_results: {weight_results}\n")
                out_file.write(f"  theta_results: {theta_results}\n")
                out_file.write("-" * 40 + "\n")
                
            except Exception as e:
                out_file.write(f"Error reading {json_file}: {e}\n")
                out_file.write("-" * 40 + "\n")
    
    print(f"Simple format written to: {output_file}")

def read_and_write_csv_format(data_directory=".", json_pattern="f_*.json", output_file="resonator_parameters_csv.txt"):
    """
    Read all JSON files and write in CSV format for easy copying
    """
    
    # Construct full path pattern
    full_pattern = os.path.join(data_directory, json_pattern)
    
    # Find all JSON files
    json_files = glob.glob(full_pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {full_pattern}")
        return
    
    # Sort files for consistent output
    json_files.sort()
    
    with open(output_file, 'w') as out_file:
        out_file.write("CSV Format Output:\n")
        out_file.write("=" * 80 + "\n")
        out_file.write("filename,freq0,clk_freq,lf,weight_results,theta_results\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                filename = Path(json_file).stem  # filename without extension
                
                # Extract the required fields
                freq0 = data.get('freq0', 'N/A')
                clk_freq = data.get('clk_freq', 'N/A')
                lf = data.get('lf', 'N/A')
                weight_results = data.get('weight_results', [])
                theta_results = data.get('theta_results', [])
                
                # Convert lists to string representation
                weight_str = str(weight_results).replace(',', ';') if weight_results != 'N/A' else 'N/A'
                theta_str = str(theta_results).replace(',', ';') if theta_results != 'N/A' else 'N/A'
                
                out_file.write(f"{filename},{freq0},{clk_freq},{lf},\"{weight_str}\",\"{theta_str}\"\n")
                
            except Exception as e:
                out_file.write(f"Error reading {json_file}: {e}\n")
    
    print(f"CSV format written to: {output_file}")

def read_and_write_detailed(data_directory=".", json_pattern="f_*.json", output_file="resonator_parameters_detailed.txt"):
    """
    Read all JSON files and write with more details including input_freq
    """
    
    # Construct full path pattern
    full_pattern = os.path.join(data_directory, json_pattern)
    
    # Find all JSON files
    json_files = glob.glob(full_pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {full_pattern}")
        return
    
    # Sort files for consistent output
    json_files.sort()
    
    with open(output_file, 'w') as out_file:
        out_file.write(f"Found {len(json_files)} JSON files\n")
        out_file.write("=" * 80 + "\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                filename = Path(json_file).name
                
                # Extract the required fields
                input_freq = data.get('input_freq', 'N/A')
                freq0 = data.get('freq0', 'N/A')
                clk_freq = data.get('clk_freq', 'N/A')
                lf = data.get('lf', 'N/A')
                lp = data.get('lp', 'N/A')
                weight_results = data.get('weight_results', 'N/A')
                theta_results = data.get('theta_results', 'N/A')
                iterations = data.get('iterations', 'N/A')
                mse_mean = data.get('mse_mean', 'N/A')
                
                out_file.write(f"File: {filename}\n")
                out_file.write(f"  input_freq: {input_freq}\n")
                out_file.write(f"  freq0: {freq0}\n")
                out_file.write(f"  clk_freq: {clk_freq}\n")
                out_file.write(f"  lf: {lf}\n")
                out_file.write(f"  lp: {lp}\n")
                out_file.write(f"  weight_results: {weight_results}\n")
                out_file.write(f"  theta_results: {theta_results}\n")
                out_file.write(f"  iterations: {iterations}\n")
                out_file.write(f"  mse_mean: {mse_mean}\n")
                out_file.write("-" * 40 + "\n")
                
            except Exception as e:
                out_file.write(f"Error reading {json_file}: {e}\n")
                out_file.write("-" * 40 + "\n")
    
    print(f"Detailed format written to: {output_file}")

def write_all_formats_to_single_file(data_directory=".", json_pattern="f_*.json", output_file="all_resonator_parameters.txt"):
    """
    Write all three formats to a single file for convenience
    """
    
    # Construct full path pattern
    full_pattern = os.path.join(data_directory, json_pattern)
    
    # Find all JSON files
    json_files = glob.glob(full_pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {full_pattern}")
        return
    
    # Sort files for consistent output
    json_files.sort()
    
    with open(output_file, 'w') as out_file:
        out_file.write("RESONATOR PARAMETERS ANALYSIS\n")
        out_file.write("=" * 80 + "\n")
        out_file.write(f"Found {len(json_files)} JSON files\n")
        out_file.write(f"Data directory: {data_directory}\n")
        out_file.write("=" * 80 + "\n\n")
        
        # Section 1: CSV Format (most compact)
        out_file.write("1. CSV FORMAT (for spreadsheet import):\n")
        out_file.write("=" * 50 + "\n")
        out_file.write("filename,input_freq,freq0,clk_freq,lf,lp,weight_results,theta_results,mse_mean,iterations\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                filename = Path(json_file).stem
                input_freq = data.get('input_freq', 'N/A')
                freq0 = data.get('freq0', 'N/A')
                clk_freq = data.get('clk_freq', 'N/A')
                lf = data.get('lf', 'N/A')
                lp = data.get('lp', 'N/A')
                weight_results = data.get('weight_results', [])
                theta_results = data.get('theta_results', [])
                mse_mean = data.get('mse_mean', 'N/A')
                iterations = data.get('iterations', 'N/A')
                
                # Convert lists to string representation
                weight_str = str(weight_results).replace(',', ';') if weight_results != 'N/A' else 'N/A'
                theta_str = str(theta_results).replace(',', ';') if theta_results != 'N/A' else 'N/A'
                
                out_file.write(f"{filename},{input_freq},{freq0},{clk_freq},{lf},{lp},\"{weight_str}\",\"{theta_str}\",{mse_mean},{iterations}\n")
                
            except Exception as e:
                out_file.write(f"Error reading {json_file}: {e}\n")
        
        out_file.write("\n" + "=" * 80 + "\n\n")
        
        # Section 2: Detailed Format
        out_file.write("2. DETAILED FORMAT:\n")
        out_file.write("=" * 50 + "\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                filename = Path(json_file).name
                
                # Extract all available fields
                input_freq = data.get('input_freq', 'N/A')
                freq0 = data.get('freq0', 'N/A')
                clk_freq = data.get('clk_freq', 'N/A')
                lf = data.get('lf', 'N/A')
                lp = data.get('lp', 'N/A')
                weight_results = data.get('weight_results', 'N/A')
                theta_results = data.get('theta_results', 'N/A')
                chosen_weights = data.get('chosen_weights', 'N/A')
                chosen_bias = data.get('chosen_bias', 'N/A')
                mse_mean = data.get('mse_mean', 'N/A')
                iterations = data.get('iterations', 'N/A')
                converted_from_freq = data.get('converted_from_freq', 'N/A')
                
                out_file.write(f"File: {filename}\n")
                out_file.write(f"  input_freq: {input_freq}\n")
                out_file.write(f"  freq0: {freq0}\n")
                out_file.write(f"  clk_freq: {clk_freq}\n")
                out_file.write(f"  lf: {lf}\n")
                out_file.write(f"  lp: {lp}\n")
                out_file.write(f"  weight_results: {weight_results}\n")
                out_file.write(f"  theta_results: {theta_results}\n")
                out_file.write(f"  chosen_weights: {chosen_weights}\n")
                out_file.write(f"  chosen_bias: {chosen_bias}\n")
                out_file.write(f"  mse_mean: {mse_mean}\n")
                out_file.write(f"  iterations: {iterations}\n")
                out_file.write(f"  converted_from_freq: {converted_from_freq}\n")
                out_file.write("-" * 40 + "\n")
                
            except Exception as e:
                out_file.write(f"Error reading {json_file}: {e}\n")
                out_file.write("-" * 40 + "\n")
    
    print(f"All formats written to: {output_file}")

if __name__ == "__main__":
    # Specify your data directory
    data_path = "/home/data/lf"  # Change this to your actual path
    
    # If directory doesn't exist, use current directory
    if not os.path.exists(data_path):
        print(f"Directory {data_path} not found, using current directory")
        data_path = "."
    
    print(f"Reading JSON files from: {data_path}")
    print()
    
    # Option A: Write all formats to separate files
    print("Creating separate files for each format...")
    read_and_write_json_files(data_path, output_file="resonator_simple.txt")
    read_and_write_csv_format(data_path, output_file="resonator_csv.txt")
    read_and_write_detailed(data_path, output_file="resonator_detailed.txt")
    
    # Option B: Write everything to one file
    print("Creating combined file...")
    write_all_formats_to_single_file(data_path, output_file="all_resonator_data.txt")
    
    print("\nAll files created successfully!")
    print("Files created:")
    print("  - resonator_simple.txt          (simple format)")
    print("  - resonator_csv.txt             (CSV format for spreadsheet)")
    print("  - resonator_detailed.txt        (detailed format)")
    print("  - all_resonator_data.txt        (all formats combined)")
