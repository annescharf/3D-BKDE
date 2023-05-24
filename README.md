# 3D Binned Kernel Density Estimation
GitHub repository: https://github.com/kulits2/3D-BKDE

<img width="30%" alt="An example output animation." src="out.gif">

## Description
This app provides a quick way to estimate and visualize 3D space use.

## Documentation
This app takes a `Trajectory`, interpolates it to a fixed time frequency, bins it into a 3D histogram, and convolves a gaussian kernel over the histogram to produce an estimate for 3D space use. It filters the convolved data to remove voxels that contribute least to the total space use. It optionally downloads `SRTM1` data with the [elevation](https://github.com/bopen/elevation) package and displays it underneath the voxels.

### Input data
A pre-filtered MovingPandas `TrajectoryCollection` in Movebank format. Each `Trajectory` must have a valid non-zero height above ellipsoid.

### Output data
The input MovingPandas `TrajectoryCollection` is returned.

### Artefacts
For each `Trajectory` in the input `TrajectoryCollection` two outputs are produced:

`{n}.png`: An image visualization of the space use.

`{n}.gif`: An animated visualization of the space use.

### Settings
`Resampling Frequency` (`freq`): The time frequency of track resampling before binning. Accepts any `unit` value at least one second from https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html

`Maximum Segment Length` (`max_seg_len`): Remove trajectory segments with greater than this time between fixes. Accepts any `unit` value at least one second from https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html

`Z Column Name` (`z_col`): The name of the column that stores the HAE.

`Kernel Radius` (`kernel_radius`): THe radius of the kernel in grid units.

`Kernel Sigma` (`kernel_sigma`): The gaussian kernel sigma.

`Latitude Grid Dimension` (`lat_grid_dim`): The number of latitudinal grid cells.

`Longitude Grid Dimension` (`lon_grid_dim`): The number of longitudinal grid cells.

`Altitude Grid Dimension` (`alt_grid_dim`): The number of altitudinal grid cells.

`KDE Percentile` (`percentile`): Display the most frequently occupied voxels that make up the given percent of the total time spent.

`Show Topography` (`show_topo`): Display a topographic map from SRTM1_ELLIP underneath the voxels.

### Null or error handling
If the app runs out of memory, try again with smaller grid dimensions, higher `freq` (do not use `1s` if your fixes are sampled at `1h`), or filter out more fixes before running the app.