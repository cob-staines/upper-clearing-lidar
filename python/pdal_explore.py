datapath = """C:\\Users\\Cob\\Documents\\educational\\usask\\research\\data\\LiDAR\\19_045\\19_045_grid_60m_WGS84_utm11N_nocolor - Scanner 1 - 190214_190320_Scanner_1 - originalpoints.las"""

json = """
{
  "pipeline": [
    "C:/Users/Cob/Documents/educational/usask/research/data/LiDAR/19_045/19_045_grid_60m_WGS84_utm11N_nocolor - Scanner 1 - 190214_190320_Scanner_1 - originalpoints.las",
    {
        "type": "filters.sort",
        "dimension": "X"
    }
  ]
}"""

json = """
{
    "pipeline": [
        "C:/Users/Cob/Documents/educational/usask/research/data/LiDAR/19_045/19_045_grid_60m_WGS84_utm11N_nocolor - Scanner 1 - 190214_190320_Scanner_1 - originalpoints.las",
        "C:/Users/Cob/Documents/educational/usask/research/data/LiDAR/19_045/outpit.las",
    ]
}"""

json = """
[
    "C:/Users/Cob/Documents/educational/usask/research/data/LiDAR/19_045/19_045_grid_60m_WGS84_utm11N_nocolor - Scanner 1 - 190214_190320_Scanner_1 - originalpoints.las",
    "C:/Users/Cob/Documents/educational/usask/research/data/LiDAR/19_045/output.las",
]
"""

import pdal
pipeline = pdal.Pipeline(json)
pipeline.validate() # check if our JSON and options were good
pipeline.loglevel = 8 #really noisy
count = pipeline.execute()
arrays = pipeline.arrays
metadata = pipeline.metadata
log = pipeline.log
