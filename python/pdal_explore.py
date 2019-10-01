datapath = """C:\\Users\\Cob\\Documents\\educational\\usask\\research\\data\\LiDAR\\esc\\19_052_a_2_120m_grid_WGS84_utm_N11_nocolor - Scanner 1 - 190221_181450_Scanner_1 - originalpoints.las"""

json = """
{
  "pipeline": [
    "C:/Users/Cob/Documents/educational/usask/research/data/LiDAR/esc/19_052_a_2_120m_grid_WGS84_utm_N11.las",
    {
        "type": "filters.sort",
        "dimension": "X"
    }
  ]
}"""

import pdal
pipeline = pdal.Pipeline(json)
pipeline.validate() # check if our JSON and options were good
pipeline.loglevel = 8 #really noisy
count = pipeline.execute()
arrays = pipeline.arrays
metadata = pipeline.metadata
log = pipeline.log
