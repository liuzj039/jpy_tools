import dataclasses
from typing import ClassVar, Any, Optional, Dict

import seaborn.objects as so
import numpy as np
import matplotlib as mpl
from matplotlib.artist import Artist
from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableColor,
    MappableFloat,
    MappableStyle,
    resolve_properties,
    resolve_color,
    document_properties,
)
from collections import defaultdict

from seaborn._core.scales import Scale
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat
from seaborn._core.typing import Vector
import pandas as pd


class AxlineBase(so.Path):
    """
    Abstract ancestor for the Axline marks.
    See also:
    ---------
    Axline : Arbitrary line mark.
    Axhline : Horizontal line mark.
    Axvline : Vertical line mark.
    """
    _sort: ClassVar[bool] = False

    def _get_passthrough_points(self, data: dict):
        raise NotImplementedError()

    def _plot(self, split_gen, scales, orient):

        for keys, data, ax in split_gen():
            vals = resolve_properties(self, keys, scales)

            # Use input data, because custom aesthetics are not supported, yet, 
            # and x/y values are not provided in `vals`. Later on, I would like
            # to utilize `intercept` and `slope` instead.
            if not "x" in vals and "x" in data.columns:
                vals["x"] = data["x"]
            if not "y" in vals and "y" in data.columns:
                vals["y"] = data["y"]

            vals["color"] = resolve_color(self, keys, scales=scales)
            vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
            vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

            artist_kws = self.artist_kws.copy()
            xy1, xy2 = self._get_passthrough_points(vals)

            for point1, point2 in zip(xy1, xy2):
                ax.axline(
                    point1,
                    point2,
                    color=vals["color"],
                    linewidth=vals["linewidth"],
                    linestyle=vals["linestyle"],
                    marker=vals["marker"],
                    markersize=vals["pointsize"],
                    markerfacecolor=vals["fillcolor"],
                    markeredgecolor=vals["edgecolor"],
                    markeredgewidth=vals["edgewidth"],
                    **artist_kws,
                )

@dataclasses.dataclass
class Axline(AxlineBase):
    """
    A mark adding *arbitrary* line to your plot.
    
    TODO: MAPPING NOT SUPPORTED YET.
    At this phase, we're able to use it with scalars only. 
    The structure is prepared, but there is a bug (or a feature)
    that limit's used scales to predefined aesthetics only. And 
    neither `intercept` nor `slope` is among them.
    Hotfix would utilize e.g. `x` instead of `intercept` and `y`
    instead of `slope`. BUT I won't to that here because it would
    be too confusing later on. Instead I'll postpone resolving 
    this issue until there is possibility to use own aesthetics.
    See also
    --------
    Axhline : A mark adding *horizontal* line to your plot.
    Axvline : A mark adding *vertical* line to your plot.
    Examples
    --------
    .. include:: ../docstrings/objects.Axline.rst    # TODO: Add
    """
    intercept: MappableFloat = Mappable(0)
    slope: MappableFloat =Mappable(1)

    def _get_passthrough_points(self, vals: dict):
        if not hasattr(vals["intercept"], "__iter__"):
            vals["intercept"] = [vals["intercept"]]
        if not hasattr(vals["slope"], "__iter__"):
            vals["slope"] = [vals["slope"]]

        xy1 = [(0, intercept) for intercept in vals["intercept"]]
        xy2 = [(1, intercept + slope) for intercept, slope in zip(vals["intercept"], vals["slope"])]
        return xy1, xy2


@dataclasses.dataclass
class Axhline(AxlineBase):
    """
    A mark adding *horizontal* line to the plot.
    See also
    --------
    Axline : A mark adding *arbitrary* line to the plot.
    Axvline : A mark adding *vertical* line to the plot.
    Examples
    --------
    .. include:: ../docstrings/objects.Axhline.rst    # TODO: Add
    """

    y: MappableFloat = Mappable(0)

    def _get_passthrough_points(self, vals: dict):
        if not hasattr(vals["y"], "__iter__"):
            vals["y"] = [vals["y"]]
        xy1 = ((0, y) for y in  vals["y"])
        xy2 = ((1, y) for y in  vals["y"])
        return xy1, xy2


@dataclasses.dataclass
class Axvline(AxlineBase):
    """
    A mark adding *vertical* line to the plot.
    See also
    --------
    Axline : A mark adding arbitrary line to the plot.
    Axhline : A mark adding horizontal line to the plot.
    Examples
    --------
    .. include:: ../docstrings/objects.Axvline.rst    # TODO: Add
    """
    x: MappableFloat = Mappable(0)

    def _get_passthrough_points(self, vals: dict):
        if not hasattr(vals["x"], '__iter__'):
            vals["x"] = [vals["x"]]
        xy1 = ((x, 0) for x in vals["x"])
        xy2 = ((x, 1) for x in vals["x"])
        return xy1, xy2


class BarBase(so.Mark):
    def _make_patches(self, data, scales, orient):
        kws = self._resolve_properties(data, scales)
        if orient == "x":
            kws["x"] = (data["x"] - data["width"] / 2).to_numpy()
            kws["y"] = data["baseline"].to_numpy()
            kws["w"] = data["width"].to_numpy()
            kws["h"] = (data["y"] - data["baseline"]).to_numpy()
        else:
            kws["x"] = data["baseline"].to_numpy()
            kws["y"] = (data["y"] - data["width"] / 2).to_numpy()
            kws["w"] = (data["x"] - data["baseline"]).to_numpy()
            kws["h"] = data["width"].to_numpy()

        kws.pop("width", None)
        kws.pop("baseline", None)

        val_dim = {"x": "h", "y": "w"}[orient]
        bars, vals = [], []

        for i in range(len(data)):
            row = {k: v[i] for k, v in kws.items()}

            # Skip bars with no value. It's possible we'll want to make this
            # an option (i.e so you have an artist for animating or annotating),
            # but let's keep things simple for now.
            if not np.nan_to_num(row[val_dim]):
                continue

            bar = mpl.patches.Rectangle(
                xy=(row["x"], row["y"]),
                width=row["w"],
                height=row["h"],
                facecolor=row["facecolor"],
                edgecolor=row["edgecolor"],
                linestyle=row["edgestyle"],
                linewidth=row["edgewidth"],
                **self.artist_kws,
            )
            bars.append(bar)
            vals.append(row[val_dim])

        return bars, vals

    def _resolve_properties(self, data, scales):
        resolved = resolve_properties(self, data, scales)

        resolved["facecolor"] = resolve_color(self, data, "", scales)
        resolved["edgecolor"] = resolve_color(self, data, "edge", scales)

        fc = resolved["facecolor"]
        if isinstance(fc, tuple):
            resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
        else:
            fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
            resolved["facecolor"] = fc

        return resolved

    def _legend_artist(
        self,
        variables,
        value: Any,
        scales,
    ) -> Artist:
        # TODO return some sensible default?
        key = {v: value for v in variables}
        key = self._resolve_properties(key, scales)
        artist = mpl.patches.Patch(
            facecolor=key["facecolor"],
            edgecolor=key["edgecolor"],
            linewidth=key["edgewidth"],
            linestyle=key["edgestyle"],
        )
        return artist


@dataclasses.dataclass
class Box(BarBase):
    """
    An oriented rectangular mark drawn between min/max values.
    Examples
    --------
    .. include:: ../docstrings/objects.Box.rst
    """

    color: MappableColor = Mappable("C0")
    alpha: MappableFloat = Mappable(0.7)
    fill: MappableBool = Mappable(True)
    edgecolor: MappableColor = Mappable(depend="color")
    edgealpha: MappableFloat = Mappable(1)
    edgewidth: MappableFloat = Mappable(rc="patch.linewidth")
    edgestyle: MappableFloat = Mappable("-")
    width: MappableFloat = Mappable(0.8, grouping=False)

    def _clip_edges(self, artist: Artist, ax):
        """Add a clip path to patch artist so that edges do not extend past face."""
        # This is a hack to handle the fact that the edge lines are centered on
        # the actual extents of the patch and overlap when stacking or dodging.
        # We may discover that this causes issues and needs to be revisited.

        # Because we are clipping, the edges end up looking half as wide as they
        # actually are. I don't love this clumsy workaround, which is going to
        # cause surprises if you work with the artists directly.
        artist.set_linewidth(artist.get_linewidth() * 2)
        linestyle = artist.get_linestyle()
        if linestyle[1]:
            linestyle = (linestyle[0], tuple(x / 2 for x in linestyle[1]))
        artist.set_linestyle(linestyle)

        # It should be faster to clip with a bbox than a path, but I cant't work
        # out how to get the intersection with the axes bbox.
        artist.set_clip_path(artist.get_path(), artist.get_transform() + ax.transData)
        if self.artist_kws.get("clip_on", True):
            # It seems the above hack undoes the default axes clipping
            artist.set_clip_box(ax.bbox)

    def _plot(self, split_gen, scales, orient):
        patches = defaultdict(list)

        for keys, data, ax in split_gen():
            kws = {}

            resolved = self._resolve_properties(keys, scales)

            kws["facecolor"] = resolved["facecolor"]
            kws["edgecolor"] = resolved["edgecolor"]
            kws["linewidth"] = resolved["edgewidth"]
            kws["linestyle"] = resolved["edgestyle"]

            other = {"x": "y", "y": "x"}[orient]

            if not set(data.columns) & {f"{other}min", f"{other}max"}:
                agg = {f"{other}min": (other, "min"), f"{other}max": (other, "max")}
                data = data.groupby([orient, "width"]).agg(**agg).reset_index()

            for row in data.itertuples():
                if orient == "x":
                    verts = np.array(
                        [
                            row.x - row.width / 2,
                            row.ymin,
                            row.x + row.width / 2,
                            row.ymin,
                            row.x + row.width / 2,
                            row.ymax,
                            row.x - row.width / 2,
                            row.ymax,
                        ]
                    ).reshape((4, 2))
                else:
                    verts = np.array(
                        [
                            row.xmin,
                            row.y - row.width / 2,
                            row.xmax,
                            row.y - row.width / 2,
                            row.xmax,
                            row.y + row.width / 2,
                            row.xmin,
                            row.y + row.width / 2,
                        ]
                    ).reshape((4, 2))

                patches[ax].append(mpl.patches.Polygon(verts, **kws))
                ax.update_datalim(verts)

        for ax, ax_patches in patches.items():
            for patch in ax_patches:
                self._clip_edges(patch, ax)
                ax.add_patch(patch)


@dataclasses.dataclass
class KdeDensityColor(Stat):
    bins: int = 20
    vmax: Optional[float] = None

    def getKdeDensity(self, data: pd.DataFrame) -> pd.DataFrame:
        from scipy.interpolate import interpn
        from scipy.stats import gaussian_kde
        bins = self.bins
        x = data['x'].values
        y = data['y'].values
        if bins > 0:
            data_, x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
            z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data_ , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
            z[np.where(np.isnan(z))] = 0.0
        else:
             xy = np.vstack([x,y])
             z = gaussian_kde(xy)(xy)

        data['color1'] = z
        if self.vmax is None:
            vmax = data['color1'].max()
        else:
            vmax = self.vmax
        data['color1'] = data['color1'].map(lambda _: min(_, vmax))
        return data
    
    def __call__(self, data: pd.DataFrame, groupby: GroupBy, orient:str, scales: Dict[str, Scale]) -> pd.DataFrame:
        res = (
            groupby
            .apply(data, self.getKdeDensity)
            .sort_values('color1')
        )
        res['color'] = res['color1']
        res = res.drop(columns=['color1'])
        return res