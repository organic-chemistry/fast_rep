import pandas as pd
import numpy as np
import time
import ast
def parse_regions(region_str):
    regions = []
    for part in region_str.split(","):
        if ":" in part:
            chrom, coords = part.split(":")
            start, end = map(int, coords.split("-"))
            regions.append([chrom, start, end])
        else:
            regions.append([part, 0, None])
    return regions



def load_bed_data_regions(root, regions_str=None, column_spec="signal",sep="\t"):
    """
    Load BED/bedGraph data with coordinates and signals, optionally filtered by regions.
    
    Args:
        root (str): Path to BED or bedGraph file.
        regions_str (str, optional): Regions in "chrom:start-end" format, comma-separated.
        column_spec (str or int, optional or list): Column name or index to use as signal. Default is "signal".
        if column_spec is str or int the signals wil be directly in data[chrom_str]['signals'] 
        if column_spec is a list then all data[chrom_str]['signals']  will be a dictionnary 
    
    Returns:
        tuple: (data dictionary (region_key: data), resolution in kilobases)
    """
    # Determine if file has a header by checking if first line starts with #
    skip = 0
    meta = {}
    with open(root, 'r') as f:
        first_line = f.readline().strip()
        has_header = first_line.startswith("#")

        if has_header:
            headers = first_line.lstrip('#').split(sep)
            skip +=1
        second_line = f.readline().strip()
        has_meta= second_line.startswith("#")
        if has_meta:
            meta = second_line.lstrip('#')
            meta = ast.literal_eval(meta)
            skip +=1

    
    # Reset file pointer
    if has_header:
        df = pd.read_csv(root, sep=sep, skiprows=skip, names=headers)
    else:
        df = pd.read_csv(root, sep=sep, names=["chrom", "start", "end", "signal"])
        headers = ["chrom", "start", "end", "signal"]
    # Handle column selection

    if type(column_spec) != list:
        return_signal = True
        column_specs = [column_spec]
    else:
        return_signal = False
        column_specs = column_spec

    try:
        # If column_spec is an integer, use it as an index
        column_names = [headers[int(column_spec)] for column_spec in column_specs]
    except ValueError:
        # If column_spec is a string, use it as a column name
        column_names = column_specs
        for column_name in  column_names:
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in {root}")
    
    # Process regions if specified
    data = {}
    resolutions = []
    
    if regions_str:
        regions = parse_regions(regions_str)
        
        for i in range(len(regions)):

            chrom_region, start_region, end_region = regions[i]
            if end_region == None:
                end_region = max(df['end'][df['chrom'] == chrom_region])
            regions[i][2] = end_region

            region_key = f"{chrom_region}:{start_region}-{end_region}"
            
            # Filter rows overlapping with this region
            mask = (df['chrom'] == chrom_region) & \
                  (df['start'] < end_region) & \
                  (df['end'] > start_region)
            region_df = df[mask]
            
            # Extract start, end, and signal arrays
            start = region_df['start'].values
            ends = region_df['end'].values
            if len(column_names) == 1 and return_signal:
                signals = region_df[column_name].fillna(0).values.astype(np.float64)
            else:
                signals ={column_name:region_df[column_name].fillna(0).values.astype(np.float64) for column_name in column_names}
            
            # Store in data dictionary
            data[region_key] = {
                'chrom': chrom_region,
                'start': start,
                'end': ends,
                'signals': signals
            }
            
            # Calculate resolution for this region
            if len(start) >= 2:
                res = start[1] - start[0]
            else:
                res = 1  # Default if no data
            resolutions.append(res)
    else:
        # Process all chromosomes
        for chrom in df['chrom'].unique():
            chrom_str = str(chrom)
            chrom_df = df[df['chrom'] == chrom]
            
            start = chrom_df['start'].values
            ends = chrom_df['end'].values
            if len(column_names) == 1 and return_signal:
                signals = chrom_df[column_name].fillna(0).values.astype(np.float64)
            else:
                signals ={column_name:chrom_df[column_name].fillna(0).values.astype(np.float64) for column_name in column_names}
            
            data[chrom_str] = {
                'chrom': chrom_str,
                'start': start,
                'end': ends,
                'signals': signals
            }
            
            # Calculate resolution for this chromosome
            if len(start) >= 2:
                res = start[1] - start[0]
            else:
                res = 1
            resolutions.append(res)
    
    # Check resolution consistency and use the first one
    if resolutions:
        if len(set(resolutions)) > 1:
            # Just issue a warning instead of error for flexibility
            warnings.warn("Inconsistent resolutions across regions")
        resolution = resolutions[0]
    else:
        resolution = 1

    #print(data)
    for region in data.keys():
        if len(data[region]["start"])==0:
            print(f"Empty data for region {region}")
            print(f"Chromosome available: {str(set(df.chrom))}")
    
    return data, resolution / 1000 , meta # Return in kilobases

def load_muli_from_bedGraph(root, regions_str=None,column_specs=[]):
    """
    Load bedGraph data with coordinates and signals, optionally filtered by regions.
    
    Args:
        root (str): Path to bedGraph file.
        regions_str (str, optional): Regions in "chrom:start-end" format. or "chrom"
    
    Returns:
        tuple: (RFD dictionary (chrom/region: data), resolution in kilobases)
    """
    # Load data
    return load_bed_data_regions(root, regions_str,column_specs)

def load_RFD_from_bedGraph(root, regions_str=None):
    """
    Load bedGraph data with coordinates and signals, optionally filtered by regions.
    
    Args:
        root (str): Path to bedGraph file.
        regions_str (str, optional): Regions in "chrom:start-end" format. or "chrom"
    
    Returns:
        tuple: (RFD dictionary (chrom/region: data), resolution in kilobases)
    """
    # Load data
    return load_bed_data_regions(root, regions_str)


def write_bedgraph(output_path, data_dict, resolution=None):
    """
    Write processed data into a bedGraph file.
    
    Args:
        output_path (str): Path to output bedGraph file.
        data_dict (dict): Data dictionary from `load_RFD_from_bedGraph`.
        resolution (float): Resolution in kilobases (optional, for logging).
    """
    with open(output_path, "w") as f:
        for key in data_dict:
            chrom = data_dict[key]['chrom']
            start = data_dict[key]['start']
            end = data_dict[key]['end']
            signals = data_dict[key]['signals']
            
            # Write each interval
            for s, e, sig in zip(start, end, signals):
                f.write(f"{chrom}\t{s}\t{e}\t{sig:.6f}\n")
    
    print(f"BedGraph written to {output_path} with resolution {resolution} Kb.")

def load_bed_data(bed_path, chromosome, start, end, resolution_kb, column_spec):
    """
    Load a BED file and extract a specific column as the signal.
    
    Args:
        bed_path (str): Path to the BED file.
        chromosome (str): Chromosome name (e.g., 'chr1').
        start (int): Start position (base pairs).
        end (int): End position (base pairs).
        resolution_kb (float): Resolution in kilobases.
        column_spec (str or int): Column name (if header exists) or index (0-based).
    Returns:
        x (array): Genomic positions (base pairs).
        y (array): Signal values.
        column_name (str): Name of the column used.
    """
    # Read the first line (comment) to get the header names
    region = f"{chromosome}:{start}-{end}"
    data,res,meta = load_bed_data_regions(bed_path,regions_str=region,column_spec=column_spec)

    # Calculate midpoint and bin
    """
    df['midpoint'] = (df['start'] + df['end']) / 2
    df['bin'] = (df['midpoint'] // resolution_bp).astype(int)

    # Extract signal values
    binned = df.groupby('bin')[column].mean().reset_index()

    #print(df.head())

    # Create x and y arrays
    # Generate x-axis in base pairs
    x = np.arange(start, end, resolution_bp)
    y = np.zeros(len(x))
    y[binned['bin']] = binned[column]
    """
    #return x, y, column_name
    return data[region]["start"],data[region]["signals"], column_spec,meta

def write_custom_bedgraph(output_path, data_dict, resolution=None,meta={}):
    """
    Write a custom bedGraph-like file with multiple signals, including a header,
    and eventually a meta. 
    
    Args:
        output_path (str): Path to output file.
        data_dict (dict): Data dictionary with multiple signals.
        resolution (float): Resolution in kilobases (for logging).
    """
    with open(output_path, "w") as f:
        # Write header with column names
        if data_dict:
            # Get signal names from the first region's signals
            first_key = next(iter(data_dict.keys()))
            signal_names = list(data_dict[first_key]['signals'].keys())
            header = f"#chrom\tstart\tend\t" + "\t".join(signal_names) + "\n"
            f.write(header)
            if meta != {}:
                f.write(f"#{str(meta)}"+"\n")

        
        # Write data rows
        for key in data_dict:
            chrom = data_dict[key]['chrom']
            start = data_dict[key]['start']
            end = data_dict[key]['end']
            signals = data_dict[key]['signals']  # Dictionary of signals
            
            for i in range(len(start)):
                s = start[i]
                e = end[i]
                sig_values = [str(signals[name][i]) for name in signals]
                line = f"{chrom}\t{s}\t{e}\t" + "\t".join(sig_values) + "\n"
                f.write(line)
    
    print(f"Custom bedGraph written to {output_path} with resolution {resolution} Kb")


def write_custom_bedgraph_pandas(output_path, data_dict, resolution=None,meta={}):
    if not data_dict:
        raise ValueError("data_dict is empty")

    # Extract signal names
    first_key = next(iter(data_dict.keys()))
    signal_names = list(data_dict[first_key]['signals'].keys())

    # --- OPTIMIZED ROWS PREPARATION ---
    start_time = time.time()
    dfs = []  # Collect DataFrames instead of rows
    for key in data_dict:
        chrom = data_dict[key]['chrom']
        start = data_dict[key]['start']
        end = data_dict[key]['end']
        signals = data_dict[key]['signals']
        
        # Validate signal lengths
        expected_length = len(start)
        if any(len(signals[name]) != expected_length for name in signal_names):
            raise ValueError("Mismatched signal lengths in data_dict")

        # Create a DataFrame for this region (vectorized)
        region_df = pd.DataFrame({
            'chrom': [chrom] * expected_length,
            'start': start,
            'end': end,
        })
        for name in signal_names:
            region_df[name] = signals[name]
        
        dfs.append(region_df)
    
    # Concatenate all region DataFrames (faster than building rows)
    df = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - start_time
    print(f"Rows preparation + DataFrame creation: {elapsed:.4f} seconds")

    # --- CSV WRITING ---
    start_time = time.time()
    # Write DataFrame without header
    df.to_csv(
        output_path,
        sep="\t",
        header=False,
        index=False,
    )
    elapsed = time.time() - start_time
    print(f"CSV writing: {elapsed:.4f} seconds")

    # --- MANUALLY ADD COMMENTED HEADER ---
    with open(output_path, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        header = f"#chrom\tstart\tend\t" + "\t".join(signal_names) + "\n"
        if meta != {}:
            header += f"#{str(meta)}"+"\n"
        f.write(header + content)
    
    print(f"Custom bedGraph written to {output_path} with resolution {resolution} Kb")
