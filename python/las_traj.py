# las_traj takes in an las_file and a corresponding trajectory file. The function then:
#   -> merges files on gps_time
#   -> interpolates trajectory to las_points
#   -> calculates angle_from_nadir
#   -> calculates distance_to_target


def las_traj(las_in, traj_in):
    # dependencies
    import laspy
    import pandas as pd
    import numpy as np

    # import las_in
    inFile = laspy.file.File(las_in, mode="r")
    # select attributes
    las_data = pd.DataFrame({'gps_time': inFile.gps_time,
                             'x': inFile.x,
                             'y': inFile.y,
                             'z': inFile.z,
                             'intensity': inFile.intensity})
    laspy_header = inFile.header
    inFile.close()
    las_data = las_data.assign(las=True)

    # import trajectory
    traj = pd.read_csv(traj_in)
    # rename columns for consistency
    traj = traj.rename(columns={'Time[s]': "gps_time",
                                'Easting[m]': "easting_m",
                                'Northing[m]': "northing_m",
                                'Height[m]': "height_m",
                                'Roll[deg]': "roll_deg"})

    # throw our pitch, roll, yaw (at least until needed later...)
    traj = traj[['gps_time', 'easting_m', 'northing_m', 'height_m']]
    traj = traj.assign(las=False)

    # resample traj to las gps times and interpolate

    # outer merge las and traj on gps_time

    # proper merge takes too long, instead keep track of index
    outer = las_data[['gps_time', 'las']].append(traj, sort=False)
    outer = outer.reset_index()
    outer = outer.rename(columns={"index": "index_las"})

    # order by gps time
    outer = outer.sort_values(by="gps_time")
    # set index as gps_time for nearest neighbor interpolation
    outer = outer.set_index('gps_time')
    # interpolate by nearest neighbor
    interpolated = outer.interpolate(method="nearest")  # issues with other columns.... can we specify?
    # resent index for clarity

    interpolated = interpolated[interpolated['las']]
    interpolated = interpolated.sort_values(by="index_las")
    interpolated = interpolated.reset_index()
    interpolated = interpolated[['easting_m', 'northing_m', 'height_m']]

    merged = pd.concat([las_data, interpolated], axis=1)
    merged = merged.drop('las', axis=1)

    # calculate point distance from track
    p1 = np.array([merged.easting_m, merged.northing_m, merged.height_m])
    p2 = np.array([merged.x, merged.y, merged.z])
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    merged = merged.assign(distance_to_track=np.sqrt(squared_dist))

    # calculate angle from nadir
    dp = p1 - p2
    phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2])  # *180/np.pi #for degrees
    merged = merged.assign(angle_from_nadir=phi)

    return [merged, laspy_header]
