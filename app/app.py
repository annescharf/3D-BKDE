import gc
import multiprocessing
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import elevation
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyproj
import pyvista as pv
import rasterio
import shapely
from movingpandas import TrajectoryCollection
from sdk.moveapps_spec import hook_impl


@dataclass
class AppConfig:
    freq: str
    max_seg_len: str
    z_col: str
    kernel_radius: int
    kernel_sigma: float
    lat_grid_dim: int
    lon_grid_dim: int
    alt_grid_dim: int
    percentile: float
    show_topo: bool


class App(object):
    def __init__(self, moveapps_io):
        self.moveapps_io = moveapps_io

    @staticmethod
    def map_config(config: dict):
        return AppConfig(
            freq=config.get("freq", "1s"),
            max_seg_len=config.get("max_seg_len", "2h"),
            z_col=config.get("z_col", "height.above.ellipsoid"),
            kernel_radius=config.get("kernel_radius", 5),
            kernel_sigma=config.get("kernel_sigma", 1.0),
            lat_grid_dim=config.get("lat_grid_dim", 100),
            lon_grid_dim=config.get("lon_grid_dim", 100),
            alt_grid_dim=config.get("alt_grid_dim", 30),
            percentile=config.get("percentile", 95.0),
            show_topo=config.get("show_topo", True),
        )

    @hook_impl
    def execute(self, data: TrajectoryCollection, config: dict) -> TrajectoryCollection:
        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_platforms", "cpu")

        app_config = self.map_config(config=config)

        with tempfile.TemporaryFile() as input_holder:
            pickle.dump(data, input_holder)

            for traj_i in range(len(data.trajectories)):
                self.execute_once(data, app_config, traj_i)
                input_holder.seek(0)
                data = pickle.load(input_holder)

        return data

    def execute_once(self, data: TrajectoryCollection, app_config: AppConfig, traj_i: int):
        traj = data.trajectories[traj_i]

        screenshot_path = self.moveapps_io.create_artifacts_file(f"{traj_i}.png")
        gif_path = self.moveapps_io.create_artifacts_file(f"{traj_i}.gif")

        freq = pd.Timedelta(app_config.freq).to_timedelta64().astype("timedelta64[s]").astype("uint32")

        del data.trajectories, data

        traj = traj.to_crs(pyproj.CRS(4326))
        assert app_config.z_col in traj.df, f"`{app_config.z_col}` is not in the `Trajectory` object"
        traj_gdf = traj.to_line_gdf()[["prev_t", "t", app_config.z_col, "geometry"]].set_crs(4326)
        del traj
        traj_gdf = traj_gdf.loc[traj_gdf["t"] - traj_gdf["prev_t"] < pd.Timedelta(app_config.max_seg_len)]

        bbox_WGS84 = traj_gdf.total_bounds
        if app_config.show_topo:
            elevation_cache_dir = tempfile.TemporaryDirectory()
            topo_process = multiprocessing.Process(
                target=elevation.clip,
                kwargs=dict(
                    bounds=bbox_WGS84, output="DEM.tif", product="SRTM1_ELLIP", cache_dir=elevation_cache_dir.name
                ),
            )
            topo_process.start()

        if not traj_gdf["t"].is_monotonic_increasing:
            traj_gdf.sort_values("t", inplace=True)

        traj_gdf.to_crs(traj_gdf.estimate_utm_crs(), inplace=True)
        utm_crs = traj_gdf.crs

        p1 = shapely.get_point(traj_gdf.geometry, 0)
        p1_x = jnp.array(p1.x.values, dtype=jnp.float32)
        p1_y = jnp.array(p1.y.values, dtype=jnp.float32)

        p2 = shapely.get_point(traj_gdf.geometry, 1)
        p2_x = jnp.array(p2.x.values, dtype=jnp.float32)
        p2_y = jnp.array(p2.y.values, dtype=jnp.float32)

        z = jnp.array(traj_gdf[app_config.z_col], dtype=jnp.float32)

        del p1, p2

        x1, x2 = min(p1_x.min(), p2_x.min()), max(p1_x.max(), p2_x.max())
        y1, y2 = min(p1_y.min(), p2_y.min()), max(p1_y.max(), p2_y.max())
        z1, z2 = z.min(), z.max()

        period_length = np.array(traj_gdf["t"].iat[-1] - traj_gdf["prev_t"].iat[0], dtype="timedelta64[s]").astype(
            "uint32"
        )

        start_time = jnp.array(
            (traj_gdf["prev_t"] - traj_gdf["prev_t"].iat[0]).astype("timedelta64[s]"), dtype="uint32"
        )
        end_time = jnp.array((traj_gdf["t"] - traj_gdf["prev_t"].iat[0]).astype("timedelta64[s]"), dtype="uint32")

        gauss = jnp.exp(
            -0.5
            * jnp.square(
                jnp.linspace(-app_config.kernel_radius, app_config.kernel_radius, 2 * app_config.kernel_radius + 1)
            )
            / app_config.kernel_sigma**2
        )
        kernel = jnp.einsum("i,j,k->ijk", gauss, gauss, gauss)
        kernel = kernel / kernel.sum()

        del traj_gdf, gauss

        @jax.jit
        def f(p1_x, p1_y, p2_x, p2_y, z, start_time, end_time):
            times = jnp.arange(0, period_length, freq)

            start_i = start_time.searchsorted(times, side="left")
            end_i = end_time.searchsorted(times, side="right")
            valid_i = start_i == end_i
            del end_i

            i = (times - start_time[start_i]) / (end_time[start_i] - start_time[start_i])
            hist = jnp.histogramdd(
                jnp.stack(
                    [
                        p1_x[start_i] * (1 - i) + p2_x[start_i] * i,
                        p1_y[start_i] * (1 - i) + p2_y[start_i] * i,
                        z[start_i],
                    ],
                    axis=-1,
                ),
                bins=(app_config.lon_grid_dim, app_config.lat_grid_dim, app_config.alt_grid_dim),
                range=((x1, x2), (y1, y2), (z1, z2)),
                weights=valid_i,
            )[0]
            kde = jax.scipy.signal.convolve(hist, kernel, mode="same")
            sort_vals = kde.flatten().sort()
            occ = (
                kde
                > sort_vals[
                    (sort_vals.cumsum()).searchsorted(((100 - app_config.percentile) / 100) * kde.sum(), side="right")
                ]
            )
            return occ & jax.lax.pad(
                ~jnp.stack(
                    [
                        occ[:-2, 1:-1, 1:-1],
                        occ[2:, 1:-1, 1:-1],
                        occ[1:-1, :-2, 1:-1],
                        occ[1:-1, 2:, 1:-1],
                        occ[1:-1, 1:-1, :-2],
                        occ[1:-1, 1:-1, 2:],
                    ]
                ).all(axis=0),
                True,
                ((1, 1, 0), (1, 1, 0), (1, 1, 0)),
            )

        gc.collect()
        occ = f(p1_x, p1_y, p2_x, p2_y, z, start_time, end_time)
        del p1_x, p1_y, p2_x, p2_y, z, start_time, end_time, f, kernel, period_length
        gc.collect()

        etd = pv.UniformGrid()
        etd.dimensions = np.array(occ.shape) + 1
        etd.origin = (0, 0, z1)
        etd.spacing = (
            (x2 - x1) / app_config.lon_grid_dim,
            (y2 - y1) / app_config.lat_grid_dim,
            (z2 - z1) / app_config.alt_grid_dim,
        )
        etd.cell_data["occ"] = np.ravel(occ, order="F")
        etd_thresh = etd.threshold(0.5)
        del occ
        gc.collect()

        pl = pv.Plotter(off_screen=True, window_size=[2048, 2048], theme=pv.themes.DarkTheme())
        pl.add_mesh(etd_thresh, color="magenta")
        path = pl.generate_orbital_path(factor=2.5, shift=etd_thresh.length, n_points=60)
        focus = etd_thresh.center
        del etd_thresh
        if app_config.show_topo:
            try:
                topo_process.join()
                with rasterio.open(Path(elevation_cache_dir.name) / "SRTM1_ELLIP/DEM.tif") as src:
                    topo = pv.UniformGrid()
                    topo.dimensions = np.array((src.width, src.height, 1))
                    l, b, r, t = src.bounds
                    transformer = pyproj.Transformer.from_crs(4326, utm_crs, always_xy=True)
                    l_utm, b_utm = transformer.transform(l, b)
                    r_utm, t_utm = transformer.transform(r, t)
                    topo.origin = (x1 - l_utm, y1 - b_utm, 0)
                    topo.spacing = ((r_utm - l_utm) / src.width, (t_utm - b_utm) / src.height, 1)
                    topo.point_data["Elevation"] = src.read()[::-1, ::-1].flatten()
                    topo = topo.warp_by_scalar()
                elevation_cache_dir.cleanup()
                pl.add_mesh(topo, cmap="terrain")
                del topo
            except RuntimeError as e:
                print(type(e), e)
        gc.collect()

        pl.show_axes()
        pl.show_bounds(
            fmt="%.3f",
            ytitle="Latitude",
            xtitle="Longitude",
            axes_ranges=[bbox_WGS84[0], bbox_WGS84[2], bbox_WGS84[1], bbox_WGS84[3], z1, z2],
            font_size=32,
            show_zaxis=False,
            show_zlabels=False,
        )

        pl.screenshot(screenshot_path, return_img=10)

        pl.open_gif(gif_path, fps=10)
        pl.orbit_on_path(path, focus=focus, write_frames=True)
        pl.close()

        assert os.path.exists(screenshot_path)
        assert os.path.exists(gif_path)
