import re
import pandas as pd

rx_dict = {
    'picture': re.compile(r'picture\s*(?P<picture>.*)\n'),
    'threshold': re.compile(r'threshold\s*(?P<threshold>\d+)\n'),
    'ringcount_ringwidth': re.compile(r'rings\s*(?P<ringcount>\d+)\s*(?P<ringwidth>\d+,\d+).*\n'),
    'transmission': re.compile(r'transmission\s*(?P<transmission>.*)%\n'),
    'openness': re.compile(r'openness\s*(?P<openness>.*)%\n'),
    'licor_lai': re.compile(r'LiCor LAI2000\s*(?P<lai_no_cor>\d+,\d+)\s*(?P<laa_no_cor>\d+)\s*(?P<lai_s>\d+,\d+)\s*(?P<laa_s>\d+)\s*(?P<lai_cc>\d+,\d+)\s*(?P<laa_cc>\d+)\s*(?P<lai_s_cc>\d+,\d+)\s*(?P<laa_s_cc>\d+)\n')
}

def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None

def parse_file(file_in, file_out):
    """
    Parse text at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for file_object to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

    """

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    # file_object = open(filepath, 'r')
    with open(file_in, 'r') as file_object:
        # preallocate empty entry
        entry = {}

        # read first line
        line = file_object.readline()

        # for each line in file_object
        while line:
            # check for a match with regex
            key, match = _parse_line(line)

            # if match found
            if key:
                # for each match group
                for ii in range(0, match.re.groups):
                    # add group and value to entry
                    entry.update({list(match.re.groupindex.keys())[ii]: match.group(list(match.re.groupindex.keys())[ii])})

                # at the last trigger key
                if key == 'licor_lai':
                    # append entry to data
                    data.append(entry)
                    # clear entry for next round
                    entry = {}

            # read next line
            line = file_object.readline()

    # convert to df
    data = pd.DataFrame(data)
    # replace commas with decimals
    data = data.replace(',', '.', regex=True)
    # convert all columns except for first to numeric
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # write output to file
    data.to_csv(file_out, index=False)

    return data

file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\poisson\\LAI.dat"
file_out = file_in.replace('LAI.dat', 'LAI_parsed.csv')

# with open(file_in) as file:
#     file_contents = file.read()
#     print(file_contents)

lai_parsed = parse_file(file_in, file_out)
