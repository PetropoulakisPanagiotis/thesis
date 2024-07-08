input_file = './valid.txt'

querries = ['scene0568_02', 'scene0025_01', 'scene0153_00', 'scene0527_00', 'scene0086_02', 'scene0684_00', 'scene0474_04', 'scene0314_00', 'scene0558_02', 'scene0100_00', 'scene0685_01', 'scene0693_01', 'scene0077_00', 'scene0077_00', 'scene0203_01', 'scene0664_02', 'scene0553_00', 'scene0064_00', 'scene0647_00', 'scene0609_03', 'scene0574_01', 'scene0300_01', 'scene0598_02', 'scene0169_01']
for query_pattern in querries:
    # Define the query pattern
    output_file = './' + query_pattern + '.txt'

    # Read the input file and filter lines

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith(query_pattern):
                outfile.write(line)
