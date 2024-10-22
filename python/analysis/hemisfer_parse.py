# parses hemisfer "full output" csv files to extract (select) values to a flat data file

import re
import numpy as np
import pandas as pd

rx_dict = {
    # dictionary of phrases that are searched for in any arbitrary line, which may then trigger conditional searches

    'picture': re.compile(r'picture\s*(?P<picture>.*)\n'),
    'threshold': re.compile(r'threshold\s*(?P<threshold>\S+).*\n'),
    'ringcount_ringwidth': re.compile(r'rings\s*(?P<ringcount>\d+)\s*(?P<ringwidth_deg>\d+,\d+).*\n'),
    'ring_table': re.compile(r'ring\s*angle.*width\n'),
    'transmission': re.compile(r'transmission\s*(?P<transmission>.*)%\n'),
    'openness': re.compile(r'openness\s*(?P<openness>.*)%\n'),
    'corrections': re.compile(r'\scorrections\s*')
    # 'licor_lai': re.compile(r'LiCor LAI2000\s*(?P<lai_no_cor>\d+,\d+)\s*(?P<laa_no_cor>\d+)\s*(?P<lai_s>\d+,\d+)\s*(?P<laa_s>\d+)\s*(?P<lai_cc>\d+,\d+)\s*(?P<laa_cc>\d+)\s*(?P<lai_s_cc>\d+,\d+)\s*(?P<laa_s_cc>\d+)\n')

}


rx_conditional = {
    # dictionary of phrases that are searched for in conditional clauses

    'transmission_gaps': re.compile(r'\sgaps\s*(?P<transmission_gaps>.*)%\n'),
    'openness_gaps': re.compile(r'\sgaps\s*(?P<openness_gaps>.*)%\n'),
    'ring': re.compile(r'(?P<ring_num>\d)\s\S*\s\S*\s\S*\s(?P<transmission>\S*)\s(?P<transmission_s>\S*)\s(?P<contactnum>\S*)\s(?P<gaps>\S*)\s\S*\s\S*'),
    # 'ring_1': re.compile(r'1\s\S*\s\S*\s\S*\s(?P<transmission_1>\S*)\s(?P<transmission_s_1>\S*)\s(?P<contactnum_1>\S*)\s(?P<gaps_1>\S*)\s\S*\s\S*'),
    # 'ring_2': re.compile(r'2\s\S*\s\S*\s\S*\s(?P<transmission_2>\S*)\s(?P<transmission_s_2>\S*)\s(?P<contactnum_2>\S*)\s(?P<gaps_2>\S*)\s\S*\s\S*'),
    # 'ring_3': re.compile(r'3\s\S*\s\S*\s\S*\s(?P<transmission_3>\S*)\s(?P<transmission_s_3>\S*)\s(?P<contactnum_3>\S*)\s(?P<gaps_3>\S*)\s\S*\s\S*'),
    # 'ring_4': re.compile(r'4\s\S*\s\S*\s\S*\s(?P<transmission_4>\S*)\s(?P<transmission_s_4>\S*)\s(?P<contactnum_4>\S*)\s(?P<gaps_4>\S*)\s\S*\s\S*'),
    # 'ring_5': re.compile(r'5\s\S*\s\S*\s\S*\s(?P<transmission_5>\S*)\s(?P<transmission_s_5>\S*)\s(?P<contactnum_5>\S*)\s(?P<gaps_5>\S*)\s\S*\s\S*'),
    # 'ring_th_1': re.compile(r'1\s\S*\s(?P<threshold_1>\S*)\s\S*\s\S*\s(?P<transmission_1>\S*)\s(?P<transmission_s_1>\S*)\s(?P<contactnum_1>\S*)\s(?P<gaps_1>\S*)\s\S*\s\S*'),
    # 'ring_th_2': re.compile(r'2\s\S*\s(?P<threshold_2>\S*)\s\S*\s\S*\s(?P<transmission_2>\S*)\s(?P<transmission_s_2>\S*)\s(?P<contactnum_2>\S*)\s(?P<gaps_2>\S*)\s\S*\s\S*'),
    # 'ring_th_3': re.compile(r'3\s\S*\s(?P<threshold_3>\S*)\s\S*\s\S*\s(?P<transmission_3>\S*)\s(?P<transmission_s_3>\S*)\s(?P<contactnum_3>\S*)\s(?P<gaps_3>\S*)\s\S*\s\S*'),
    # 'ring_th_4': re.compile(r'4\s\S*\s(?P<threshold_4>\S*)\s\S*\s\S*\s(?P<transmission_4>\S*)\s(?P<transmission_s_4>\S*)\s(?P<contactnum_4>\S*)\s(?P<gaps_4>\S*)\s\S*\s\S*'),
    # 'ring_th_5': re.compile(r'5\s\S*\s(?P<threshold_5>\S*)\s\S*\s\S*\s(?P<transmission_5>\S*)\s(?P<transmission_s_5>\S*)\s(?P<contactnum_5>\S*)\s(?P<gaps_5>\S*)\s\S*\s\S*'),
    'licor_lai': re.compile(r'LiCor LAI2000\s*(?P<lai_no_cor>\d+,\d+)\s*(?P<laa_no_cor>\d+)\s*(?P<lai_s>\d+,\d+)\s*(?P<laa_s>\d+)\s*(?P<lai_cc>\d+,\d+)\s*(?P<laa_cc>\d+)\s*(?P<lai_s_cc>\d+,\d+)\s*(?P<laa_s_cc>\d+)\n'),
    'thimonier_lai': re.compile(r'T. & al. \(2010\)\s*(?P<lai_t_no_cor>\d+,\d+)\s*(?P<laa_t_no_cor>\d+)\s*(?P<lai_t_s>\d+,\d+)\s*(?P<laa_t_s>\d+)\s*(?P<lai_t_cc>\d+,\d+)\s*(?P<laa_t_cc>\d+)\s*(?P<lai_t_s_cc>\d+,\d+)\s*(?P<laa_t_s_cc>\d+)\n'),
    'thimonier_skyview': re.compile(r'T. & al. \(2010\)\s*(?P<skyview_no_cor>\d+([.,]\d+)?)%\s*-\s*(?P<skyview_s>\d+([.,]\d+)?)%\s*-\s*(?P<skyview_cc>\d+([.,]\d+)?)%\s*(?P<skyview_gaps_cc>\d+([.,]\d+)?)%\s*(?P<skyview_s_cc>\d+([.,]\d+)?)%\s*(?P<skyview_gaps_s_cc>\d+([.,]\d+)?)%\n')
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
            # function is terminated
    # if there are no matches, return None
    return None, None

def parse_file(file_in, file_out, hemimeta_in=None):
    """
    parses hemisfer "full output" files, returning select values as flat dataframes (csv)
    :param file_in: hemisfer "LAI.dat" file (only works for restuls exported as "full output" in hemisphere options)
    :param file_out: output csv file to be create
    :param hemimeta_in: optional metadata file for synthetic hemispheres generated with hemigen. If included, output will include hemimeta ids
        from hemimeta.csv file in same directory
    :return: data : pd.DataFrame of parsed data
    """

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    # file_object = open(file_in, 'r')
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

                # conditional lines
                if key == 'transmission':
                    line = file_object.readline()
                    match = rx_conditional['transmission_gaps'].search(line)
                    entry.update({list(match.re.groupindex.keys())[0]: match.group(list(match.re.groupindex.keys())[0])})
                elif key == 'openness':
                    line = file_object.readline()
                    match = rx_conditional['openness_gaps'].search(line)
                    entry.update({list(match.re.groupindex.keys())[0]: match.group(list(match.re.groupindex.keys())[0])})
                elif key == 'ring_table':
                    key = 'ring'
                    line = file_object.readline()
                    match = rx_conditional[key].search(line)
                    while match:
                        ring_num = match.group(list(match.re.groupindex.keys())[0])  # first match is ring num
                        for ii in range(1, match.re.groups):
                            # add group and value to entry
                            entry.update({list(match.re.groupindex.keys())[ii] + '_' + ring_num: match.group(
                                list(match.re.groupindex.keys())[ii])})
                        line = file_object.readline()
                        match = rx_conditional[key].search(line)
                # elif key == 'licor_lai':  # at the last trigger key
                #     # append entry to data
                #     data.append(entry)
                #     # clear entry for next round
                #     entry = {}
                elif key == 'corrections':  # at the last trigger key
                    # advance 4 lines
                    line = file_object.readline()
                    line = file_object.readline()
                    line = file_object.readline()
                    line = file_object.readline()

                    # licor lai
                    match = rx_conditional['licor_lai'].search(line)
                    for ii in range(0, match.re.groups):
                        # add group and value to entry
                        entry.update({list(match.re.groupindex.keys())[ii]: match.group(
                            list(match.re.groupindex.keys())[ii])})

                    line = file_object.readline()

                    # Thimonier lai
                    match = rx_conditional['thimonier_lai'].search(line)
                    for ii in range(0, match.re.groups):
                        # add group and value to entry
                        entry.update({list(match.re.groupindex.keys())[ii]: match.group(
                            list(match.re.groupindex.keys())[ii])})

                    # advance 4 lines
                    line = file_object.readline()
                    line = file_object.readline()
                    line = file_object.readline()
                    line = file_object.readline()

                    # Thimonier skyview
                    match = rx_conditional['thimonier_skyview'].search(line)
                    for ii in range(0, len(match.re.groupindex)):
                        # add group and value to entry
                        entry.update({list(match.re.groupindex.keys())[ii]: match.group(
                            list(match.re.groupindex.keys())[ii])})

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
    # data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # convert transmissions to from percent to decimal
    data[['transmission', 'transmission_gaps', 'openness', 'openness_gaps']] = data[['transmission', 'transmission_gaps', 'openness', 'openness_gaps']].astype('float') / 100

    if hemimeta_in is not None:
        # load hemimeta (metadata)
        hemimeta_in = file_in.replace('LAI.dat', 'hemimetalog.csv')
        hemimeta = pd.read_csv(hemimeta_in)
        hemimeta = hemimeta.loc[:, ['id', 'file_name']]
        hemimeta.id = hemimeta.id.astype(int)

        data = hemimeta.merge(data, how='right', left_on='file_name', right_on='picture')
        data = data.drop(columns='picture')

    # write output to file
    data.to_csv(file_out, index=False)

    return data

def main():
    # optimization sets
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\19_149\\clean\\sized\\thresholded\\LAI.dat"
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\045_052_050\\LAI_045_050_052_test.dat"
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\19_045\\clean\\sized\\thresholded\\LAI.dat"
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\19_050\\clean\\sized\\thresholded\\LAI.dat"
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\19_052\\clean\\sized\\thresholded\\LAI.dat"
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\poisson\\LAI.dat"
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\flip_rerun\\LAI.dat"
    # file_out = file_in.replace('.dat', '_parsed.dat')
    # lai_parsed = parse_file(file_in, file_out)

    # analysis sets
    # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\uf_1m_pr0_os1.01_snow_off_dem_offset.25\\outputs\\LAI.dat"
    file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\uf_1m_pr0_os0.063_snow_off_dem_offset.25_set_0\\outputs\\LAI.dat"
    # # # file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\uf_1m_pr.15_os14.5_snow_off_dem_offset.25\\outputs\\LAI.dat"
    file_out = file_in.replace('.dat', '_parsed.dat')
    lai_parsed = parse_file(file_in, file_out, hemimeta_in=True)

if __name__ == "__main__":
    main()
