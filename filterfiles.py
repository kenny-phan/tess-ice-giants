input_file = 'tesscurl_sector_70_ffic.sh'
output_file = 'filtered_70.sh'

# Open input and output files
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    # Read each line in the input file
    for line in f_in:
        # Check if the line contains '1-1'
        if '1-1' in line:
            # Write the line to the output file
            f_out.write(line)

print(f"Filtered lines containing '1-1' to {output_file}")

