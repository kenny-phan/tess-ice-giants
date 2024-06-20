input_file = '/home/ktp9/TESSNeptune24/sector42/tesscurl_sector_42_ffic.sh' 
output_file = '/home/ktp9/TESSNeptune24/sector42/filtered_42.sh'
keyword = '-2-4-'

# Open input and output files
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    # Read each line in the input file
    for line in f_in:
        # Check if the line contains the keyword
        if keyword in line:
            # Write the line to the output file
            f_out.write(line)

print(f"Filtered lines containing {keyword} to {output_file}")

