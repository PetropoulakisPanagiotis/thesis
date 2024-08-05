# read scannet valid.txt and create subsets #
input_file = './valid.txt'

querries = ['scene0568_02', 'scene0025_01', 'scene0153_00', 'scene0527_00', 'scene0086_02', 'scene0684_00', 'scene0314_00', 'scene0558_02', \
            'scene0100_02', 'scene0685_01', 'scene0693_01', 'scene0664_02', 'scene0553_00', 'scene0064_00', 'scene0647_00', 'scene0609_03', \
            'scene0574_01', 'scene0300_01', 'scene0598_02', 'scene0019_00', 'scene0063_0', 'scene0077_00', 'scene0081_00', 'scene0131_02', \
            'scene0193_01', 'scene0164_01', 'scene0221_01', 'scene0277_01', 'scene0278_01', 'scene0316_00', 'scene0338_02', 'scene0342_00', \
            'scene0356_02', 'scene0377_02', 'scene0382_01', 'scene0423_01', 'scene0432_00', 'scene0441_00', 'scene0461_00', 'scene0474_04' 
            ]

querries = set(querries)

for query_pattern in querries:
    # Define the query pattern
    output_file = './' + query_pattern + '.txt'

    # Read the input file and filter lines

    count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith(query_pattern):
                outfile.write(line)
                count += 1

    """
    if count > 1200:
        print(query_pattern)
        print(count)
        print('\n')
    """
