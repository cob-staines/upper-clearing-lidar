import re
import pandas as pd

filepath = "D:\\lenovo_2020-07-23\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\19_149\\clean\\sized\\LAI.dat"

with open(file_in) as file:
    file_contents = file.read()
    print(file_contents)

rx_dict = {
    'picture': re.compile(r'picture\\s*(?P<picture>.*)\\n')
    'threshold': re.compile(r'threshold\s*(?P<threshold>\d+)\n')
    'ringcount_ringwidth': re.compile(r'rings\s*(?P<ringcount>\d+)\s*(?P<ringwidth>\d+,\d+).\n')
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

def parse_file(filepath):
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
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            if key == 'picture':
                picture = match.group('picture')

            if key == 'threshold':
                threshold = match.group('threshold')

            if key == 'ringcount_ringwidth':
                ringcount = match.group('ringcount')
                ringwidth = match.group('ringwidth')

                row = {
                    'picture': picture,
                    'threshold': threshold,
                    'ringcount': ringcount,
                    'ringwidth': ringwidth
                }
                # append the dictionary to the data list
                data.append(row)
        #
        #     # extract school name
        #     if key == 'school':
        #         school = match.group('school')
        #
        #     # extract grade
        #     if key == 'grade':
        #         grade = match.group('grade')
        #         grade = int(grade)
        #
        #     # identify a table header
        #     if key == 'name_score':
        #         # extract type of table, i.e., Name or Score
        #         value_type = match.group('name_score')
        #         line = file_object.readline()
        #         # read each line of the table until a blank line
        #         while line.strip():
        #             # extract number and value
        #             number, value = line.strip().split(',')
        #             value = value.strip()
        #             # create a dictionary containing this row of data
        #             row = {
        #                 'School': school,
        #                 'Grade': grade,
        #                 'Student number': number,
        #                 value_type: value
        #             }
        #             # append the dictionary to the data list
        #             data.append(row)
        #             line = file_object.readline()
        #
        #     line = file_object.readline()
        #
        # # create a pandas DataFrame from the list of dicts
        # data = pd.DataFrame(data)
        # # set the School, Grade, and Student number as the index
        # data.set_index(['School', 'Grade', 'Student number'], inplace=True)
        # # consolidate df to remove nans
        # data = data.groupby(level=data.index.names).first()
        # # upgrade Score from float to integer
        # data = data.apply(pd.to_numeric, errors='ignore')
    return data

