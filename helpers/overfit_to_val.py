# read scannet valid.txt and create subsets #
input_file = './valid.txt'
out_file = './valid_slam.txt'
querries = [
'scene0664_02',
'scene0314_00',
'scene0064_00',
'scene0086_02',
'scene0598_02',
'scene0574_01',
'scene0338_02',
'scene0685_01',
'scene0300_01',
'scene0527_00',
'scene0684_00',
'scene0019_00',
'scene0193_01',
'scene0131_02',
'scene0025_01',
'scene0221_01',
'scene0164_01',
'scene0316_00',
'scene0693_01',
'scene0100_02',
'scene0609_03',
'scene0553_00',
'scene0342_00',
'scene0081_00',
'scene0278_01',
    ]   


querries = set(querries)

for query_pattern in querries:
    # Define the query pattern
    # Read the input file and filter lines

    count = 0
    with open(input_file, 'r') as infile, open(out_file, 'a') as outfile:
        for line in infile:
            if line.startswith(query_pattern):
                outfile.write(line)
                count += 1
