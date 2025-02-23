'''
Script which hosts various useful plotting functions for interpreting the input data and SysID.model results 

Created by: Jasper van Beers
'''

# ================================================================================================================================ #
# Global Imports
# ================================================================================================================================ #
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.transforms import Affine2D
from itertools import cycle
from scipy.stats import norm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as clrs
import mpl_toolkits.mplot3d.art3d as art3d
import os
import pickle as pkl
import scipy.stats as stats

# Alternative for alphashape
# https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation
try:
    import alphashape
    from descartes import PolygonPatch
    import seaborn as sns
    cmap = sns.cubehelix_palette(start=.2, rot=-.3, as_cmap=True)
except ModuleNotFoundError:
    cmap = 'viridis'
    print('[ WARNING ] Package alphashape and/or descartes not found. Some plotting utilities will not work.')


# ================================================================================================================================ #
# Functions
# ================================================================================================================================ #

def show():
    plt.show()
    return None


def prettifyAxis(ax, tight = False):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    if tight:
        plt.tight_layout()
    return None


def addVLINE(ax, x, ymin, ymax, **kwargs):
    ylim = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)
    ax.set_ylim(ylim)
    return None


def addXVSPAN(ax, xmin, xmax, **kwargs):
    xlim = ax.get_xlim()
    ax.axvspan(xmin, xmax, **kwargs)
    ax.set_xlim(xlim)
    return None



def addLegendPatch(handles, **patchKwargs):
    handles.append(mpatches.Patch(**patchKwargs))


def addLegendLine(handles, **lineKwargs):
    handles.append(mlines.Line2D([], [], **lineKwargs))


def makeFig(nrows = 1, ncolumns = 1, returnGridSpec = False, **kwargs):
    fig = plt.figure()
    if returnGridSpec:
        gs = fig.add_gridspec(nrows = nrows, ncols = ncolumns)
        return fig, gs
    else:
        return fig


def makeBoldLabel(ax, label, unit = None, which = 'x'):
    if unit is None:
        unit = ''
    else:
        unit = f', {unit}'
    axesMapper = {'x':ax.set_xlabel, 'y':ax.set_ylabel, 'z':ax.set_zlabel}
    labelList = label.split(' ')
    return None


def _SaveStatePhases(filename, states, proportion = 1, alphaShapeSensitivity = 0.1, savePath = None):
    if savePath is None:
        savePath = os.getcwd()
    statesComb = np.stack(tuple(i.reshape(-1) for i in states), axis = 1)
    stateCenters = np.vstack(([np.nanmean(i) for i in states],)*len(statesComb))
    dists = np.linalg.norm(statesComb - stateCenters, axis = 1)
    sortedIdxs = np.argsort(dists)
    sortedStates = statesComb[sortedIdxs]
    # if alphaShapeSensitivity is None:
    #     alphaShapeSensitivity = alphashape.optimizealpha(statesComb)    
    # contour = alphashape.alphashape(statesComb, alphaShapeSensitivity)
    if alphaShapeSensitivity is None:
        alphaShapeSensitivity = alphashape.optimizealpha(sortedStates[:int(proportion*len(statesComb))])    
    contour = alphashape.alphashape(sortedStates[:int(proportion*len(statesComb))], alphaShapeSensitivity)    
    with open(os.path.join(savePath, filename), 'wb') as f:
        pkl.dump(contour, f)
        f.close()
    return 
    


def _loadStatePhases(savePath):
    with open(savePath, 'rb') as f:
        contour = pkl.load(f)
    return contour



def PhasePortrait_2D(X, Y, showScatter = False, returnFig = True, numberContours = 3, firstContourPercentile = 0.5, alphaShapeSensitivity = 1.0, xLabel = None, yLabel = None, colors = None, useMedian = False):
    # Define distribution sizes for the contour lines
    partitionStep = (1 - firstContourPercentile)/numberContours
    partitions = np.arange(firstContourPercentile, 1, partitionStep)
    if partitions[-1] <= 1:
        partitions = np.hstack((partitions, np.array(1)))
    if colors is None:
        cmap = sns.color_palette("Spectral", len(partitions), as_cmap = True)
        cNorm = clrs.Normalize(vmin=0, vmax=len(partitions))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        colors = cycle([scalarMap.to_rgba(i) for i in range(len(partitions))])
    else:
        colors = cycle(colors)
    # Find center of the phase portrait
    if useMedian:
        xCenter = np.nanmedian(X)
        yCenter = np.nanmedian(Y)
    else:
        xCenter = np.nanmean(X)
        yCenter = np.nanmean(Y)
    XY = np.stack((X.reshape(-1), Y.reshape(-1)), axis = 1)
    N = len(XY)
    XYCenter = np.stack((np.ones(N)*xCenter, np.ones(N)*yCenter), axis = 1)
    dist = np.linalg.norm(XY - XYCenter, axis = 1)
    sortIdx = np.argsort(dist)
    sortedXY = XY[sortIdx]
    contours = []
    # lb = 0
    resetSensitivity = False
    for part in partitions:
        # _XY = sortedXY[lb:int(part*N)]
        _XY = sortedXY[:int(part*N)]
        # If alphaShapeSensitivity is not specified, attempt to optimize
        if alphaShapeSensitivity is None:
            alphaShapeSensitivity = alphashape.optimizealpha(_XY)
            resetSensitivity = True
        _contour = alphashape.alphashape(_XY, alphaShapeSensitivity)
        contours.append(_contour)
        if resetSensitivity:
            alphaShapeSensitivity = None
        # lb = int(part*N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if showScatter:
        ax.scatter(X, Y)
    else:
        ax.set_xlim([np.nanmin(X)*((10 - np.sign(np.nanmin(X)))/10), np.nanmax(X)*((10 + np.sign(np.nanmax(X)))/10)])
        ax.set_ylim([np.nanmin(Y)*((10 - np.sign(np.nanmin(Y)))/10), np.nanmax(Y)*((10 - np.sign(np.nanmax(Y)))/10)])
    for i, contour in enumerate(contours):
        c = next(colors)
        ax.add_patch(PolygonPatch(contour, alpha = 0.8, label = '{}-percentile contour'.format(partitions[i]), edgecolor = c, facecolor = c))
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.legend()
    if returnFig:
        return fig
    else:
        plt.show()
        return None



def PhasePortrait_2D_from_Contours(XYContour, X, Y, showScatter = False, returnFig = True, alpha = 0.8, zorder = 0, color = 'firebrick', parentFig = None, flipAxis = False):
    if parentFig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = parentFig
        ax = fig.gca()
    polygon = PolygonPatch(XYContour, alpha = alpha, edgecolor = color, facecolor = color)
    if flipAxis:
        _X = Y
        _Y = X
        X, Y = _X, _Y
        polyPath = polygon.get_path()
        polyPath._vertices = polyPath._vertices[:, [1, 0]]
        polygon.set_path(polyPath)
    ax.add_patch(polygon)
    if showScatter:
        ax.scatter(X, Y, color = color)
    else:
        ax.set_xlim([np.nanmin(X)*((10 - np.sign(np.nanmin(X)))/10), np.nanmax(X)*((10 + np.sign(np.nanmax(X)))/10)])
        ax.set_ylim([np.nanmin(Y)*((10 - np.sign(np.nanmin(Y)))/10), np.nanmax(Y)*((10 - np.sign(np.nanmax(Y)))/10)])    
    if returnFig:
        return fig
    else:
        plt.show()
        return None
        


def PhasePortrait_3D_from_2DContours(XYContour, XZContour, YZContour, X, Y, Z, alphaShapeSensitivity = 0.01, returnFig = True, alpha = 0.8, zorder = 0, color = 'firebrick', parentFig = None, label = None, xLabel = None, yLabel = None, zLabel = None, showProjections = (False, False, False), zorderProj = 20, shuffleXYZ = None):
    def _getLastCoord(coords, X, Y, Z):
        loc = np.where((X == coords[0]) & (Y == coords[1]))
        return Z[loc]

    def _getExteriorCoords(polygon, X, Y, Z):
        # Check if there is an interior
        if len(polygon.interiors):
            print('[ INFO ] FOUND AN INTERIOR. HELP!')
        # Extract coordinate exteriors
        exteriorCoords = list(polygon.exterior.coords)
        polyCoords = np.zeros((len(exteriorCoords), 3))
        extraCoords = []
        for i, eCoord in enumerate(exteriorCoords):
            polyCoords[i, 0] = eCoord[0]
            polyCoords[i, 1] = eCoord[1]
            try:
                polyCoords[i, 2] = _getLastCoord(eCoord, X, Y, Z)
            except ValueError:
                multipleZCoords = _getLastCoord(eCoord, X, Y, Z)
                polyCoords[i, 2] = multipleZCoords[0][0]
                for zCoord in range(len(multipleZCoords[0])-1):
                    extraCoords.append([eCoord[0], eCoord[1], multipleZCoords[0][zCoord + 1]])
        if len(extraCoords):
            extraCoords = np.array(extraCoords)
            polyCoords = np.vstack((polyCoords, extraCoords))
        return polyCoords

    def _extractCoordinates(contour, X, Y, Z):
        if contour.type == 'MultiPolygon':
            for i, polygon in enumerate(contour):
                if i == 0:
                    polyCoords = _getExteriorCoords(polygon, X, Y, Z)
                else:
                    _polyCoords = _getExteriorCoords(polygon, X, Y, Z)
                    polyCoords = np.vstack((polyCoords, _polyCoords))
        elif contour.type == 'Polygon':
            polyCoords = _getExteriorCoords(contour, X, Y, Z)
        else:
            raise TypeError('Expected Polygon or Multipolygon object. Got {} instead'.format(contour.type))
        polyCoords = np.array(polyCoords).reshape(-1, 3)
        return polyCoords

    def _add2DContours(ax, contour, z, zdir):
        ax.add_patch(contour)
        art3d.pathpatch_2d_to_3d(contour, z=z, zdir=zdir)
        return None


    def _getContourLimits(contour):
        if contour.type == 'MultiPolygon':
            exteriorCoordList = []
            for i, polygon in enumerate(contour):
                exteriorCoordList = exteriorCoordList + list(polygon.exterior.coords)
        elif contour.type == 'Polygon':
            exteriorCoordList = list(contour.exterior.coords)
        else:
            raise TypeError('Expected Polygon or Multipolygon object. Got {} instead'.format(contour.type))
        exteriorCoords = np.array(exteriorCoordList)
        return (np.nanmin(exteriorCoords, axis = 0), np.nanmax(exteriorCoords, axis = 0))


    # Get exterior coordinates of the alpha shape based on their contours, and infer final 
    # axis coordinate.
    # If not in order X Y Z, need to rearrange array to be conformant with X Y Z
    coordsFromXY = _extractCoordinates(XYContour, X, Y, Z)
    coordsFromXZ = _extractCoordinates(XZContour, X, Z, Y)[:, [0, 2, 1]]
    cooordFromYZ = _extractCoordinates(YZContour, Y, Z, X)[:, [2, 0, 1]]

    XYLims = _getContourLimits(XYContour)
    XZLims = _getContourLimits(XZContour)
    YZLims = _getContourLimits(YZContour)

    Xlims = (np.nanmin([XYLims[0][0], XZLims[0][0]]), np.nanmax([XYLims[1][0], XZLims[1][0]]))
    Ylims = (np.nanmin([XYLims[0][1], YZLims[0][0]]), np.nanmax([XYLims[1][1], YZLims[1][0]]))
    Zlims = (np.nanmin([XZLims[0][1], YZLims[0][1]]), np.nanmax([XZLims[1][1], YZLims[1][1]]))

    _coordinates3D = np.unique(np.vstack((coordsFromXY, coordsFromXZ, cooordFromYZ)), axis = 0)
    coordinates3D = _coordinates3D[np.where((_coordinates3D[:, 0] >= Xlims[0]) & (_coordinates3D[:, 0] <= Xlims[1])
                                            & (_coordinates3D[:, 1] >= Ylims[0]) & (_coordinates3D[:, 1] <= Ylims[1])
                                            & (_coordinates3D[:, 2] >= Zlims[0]) & (_coordinates3D[:, 2] <= Zlims[1]))]
    axisDirs = np.array(['x', 'y', 'z'])
    XYZ = np.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), axis = 1)
    Contours = np.array([XYContour, XZContour, YZContour], dtype=object)
    if shuffleXYZ is not None:
        coordinates3D = coordinates3D[:, shuffleXYZ]
        axisDirs = axisDirs[shuffleXYZ]
        XYZ = XYZ[:, shuffleXYZ]
        Contours = Contours[shuffleXYZ]
    
    # Create a new alpha figure based on the coordinates along the outer contour of the intersection
    # of the 2-D contours
    contour3D = alphashape.alphashape(coordinates3D, alphaShapeSensitivity)

    if parentFig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig = parentFig
        ax = fig.gca()

    ax.plot_trisurf(*zip(*contour3D.vertices), triangles = contour3D.faces, alpha = alpha, color = color, label = label, zorder=zorder)
    # -x, +y, -z
    # import code
    # code.interact(local=locals())
    if showProjections[0]:
        _add2DContours(ax, PolygonPatch(Contours[0], alpha = 0.5*alpha, color = color, zorder = zorderProj), np.nanmin(XYZ[:, 2])*((10 - 5*np.sign(np.nanmin(XYZ[:, 2])))/10), 'z')
        ax.set_zlim([np.nanmin(XYZ[:, 2])*((10 - 5*np.sign(np.nanmin(XYZ[:, 2])))/10), np.nanmax(XYZ[:, 2])*((10 + np.sign(np.nanmax(XYZ[:, 2])))/10)])
    if showProjections[1]:
        _add2DContours(ax, PolygonPatch(Contours[1], alpha = 0.5*alpha, color = color, zorder = zorderProj), np.nanmax(XYZ[:, 1])*((10 + 8*np.sign(np.nanmax(XYZ[:, 1])))/10), 'y')
        ax.set_ylim([np.nanmin(XYZ[:, 1])*((10 - np.sign(np.nanmin(XYZ[:, 1])))/10), np.nanmax(XYZ[:, 1])*((10 + 8*np.sign(np.nanmax(XYZ[:, 1])))/10)])
    if showProjections[2]:
        _add2DContours(ax, PolygonPatch(Contours[2], alpha = 0.5*alpha, color = color, zorder = zorderProj), np.nanmin(XYZ[:, 0])*((10 - 5*np.sign(np.nanmin(XYZ[:, 0])))/10), 'x')
        ax.set_xlim([np.nanmin(XYZ[:, 0])*((10 - 5*np.sign(np.nanmin(XYZ[:, 0])))/10), np.nanmax(XYZ[:, 0])*((10 + np.sign(np.nanmax(XYZ[:, 0])))/10)])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)

    if returnFig:
        return fig
    else:
        return None



def PhasePortrait_3D(X, Y, Z, showScatter = False, returnFig = True, contours = None, numberContours = 3, firstContourPercentile = 0.5, alphaShapeSensitivity = 1.0, xLabel = None, yLabel = None, zLabel = None, colors = None, useMedian = False, alphas = None, nSkip=None):
    # Define distribution sizes for the contour lines
    if contours is None:
        partitionStep = (1 - firstContourPercentile)/numberContours
        partitions = np.arange(firstContourPercentile, 1, partitionStep)
    else:
        numberContours = len(contours)
        firstContourPercentile = contours[0]
        partitions = contours
    if partitions[-1] <= 1:
        partitions = np.hstack((partitions, np.array(1)))
    if colors is None:
        cmap = sns.color_palette("Spectral", len(partitions), as_cmap = True)
        cNorm = clrs.Normalize(vmin=0, vmax=len(partitions))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        colors = cycle([scalarMap.to_rgba(i) for i in range(len(partitions))])
    else:
        colors = cycle(colors)
    # Specify decreasing level of transparency as contours get added, such that interior contours remain visible
    if alphas is None:
        alphas = partitions[::-1] * 0.3
    else:
        if len(alphas) + 1 < len(partitions):
            raise ValueError('Expected length of alphas to be: {}, but alphas got len(alphas) = {} instead.'.format(len(partitions)-1, len(alphas)))
        else:
            alphas = list(alphas) + [alphas[-1]*0.2]
    # Find center of the phase portrait
    if useMedian:
        xCenter = np.nanmedian(X)
        yCenter = np.nanmedian(Y)
        zCenter = np.nanmedian(Z)
    else:
        xCenter = np.nanmean(X)
        yCenter = np.nanmean(Y)
        zCenter = np.nanmean(Z)
    XYZ = np.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), axis = 1)
    if nSkip is not None:
        XYZ = XYZ[::nSkip]
    N = len(XYZ)
    XYZCenter = np.stack((np.ones(N)*xCenter, np.ones(N)*yCenter, np.ones(N)*zCenter), axis = 1)
    dist = np.linalg.norm(XYZ - XYZCenter, axis = 1)
    sortIdx = np.argsort(dist)
    sortedXYZ = XYZ[sortIdx]
    contours = []
    # lb = 0
    resetSensitivity = False
    for part in partitions:
        # _XY = sortedXY[lb:int(part*N)]
        _XYZ = sortedXYZ[:int(part*N)]
        # If alphaShapeSensitivity is not specified, attempt to optimize
        if alphaShapeSensitivity is None:
            alphaShapeSensitivity = alphashape.optimizealpha(_XYZ)
            resetSensitivity = True
        _contour = alphashape.alphashape(_XYZ, alphaShapeSensitivity)
        contours.append(_contour)
        if resetSensitivity:
            alphaShapeSensitivity = None
        # lb = int(part*N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    if showScatter:
        ax.scatter(X, Y, Z)
    colorHistory = []
    for i, contour in enumerate(contours):
        c = next(colors)
        ax.plot_trisurf(*zip(*contour.vertices), triangles = contour.faces, alpha = alphas[i], color = c)
        colorHistory.append(c)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    handles = []
    for i, c in enumerate(colorHistory):
        handles.append(mpatches.Patch(color=c, label='{}-percentile contour'.format(partitions[i])))
    ax.legend(handles = handles)
    if returnFig:
        return fig
    else:
        plt.show()
        return None



def PhasePortrait_OuterContour(X, Y, Z = None, returnFig = True, showScatter = False, alphaShapeSensitivity = 0.5, xLabel = None, yLabel = None, zLabel = None, color = 'firebrick', alpha = 0.5, cmap = cmap):
    fig = plt.figure()
    if Z is not None:
        coords = np.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), axis = 1)
        ax = fig.add_subplot(111, projection = '3d')
        if showScatter:
            ax.scatter(X, Y, Z)
    else:
        coords = np.stack((X.reshape(-1), Y.reshape(-1)), axis = 1)
        ax = fig.add_subplot(111)
        if showScatter:
            ax.scatter(X, Y)
        else:
            ax.set_xlim([np.nanmin(X)*((10 - np.sign(np.nanmin(X)))/10), np.nanmax(X)*((10 + np.sign(np.nanmax(X)))/10)])
            ax.set_ylim([np.nanmin(Y)*((10 - np.sign(np.nanmin(Y)))/10), np.nanmax(Y)*((10 - np.sign(np.nanmax(Y)))/10)])
    contour = alphashape.alphashape(coords, alphaShapeSensitivity)
    if Z is not None:
        if cmap is not None:
            ax.plot_trisurf(*zip(*contour.vertices), triangles = contour.faces, cmap = cmap)
        else:
            ax.plot_trisurf(*zip(*contour.vertices), triangles = contour.faces, alpha = alpha, color = color)
        ax.set_zlabel(zLabel)
    else:
        ax.add_patch(PolygonPatch(contour, alpha = alpha, edgecolor = color, facecolor = color))
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    # ax.legend()
    if returnFig:
        return fig
    else:
        plt.show()
        return None



def _findOuterContour(X, Y, res = 100):
    # Define bins along x
    x_bins = np.linspace(np.min(X), np.max(X), num=res)
    # Pre-allocate lower bound lines and upper bound lines of contour
    n = len(x_bins)-1
    lb = np.zeros(n)
    ub = np.zeros(n)
    for i in range(n):
        # Define current bin
        idx_bin = np.where((X <= x_bins[i+1]) & (X >= x_bins[i]))[0]
        # Find minimum Y within bin range
        idx_min = np.argmin(Y[idx_bin])
        lb[i] = idx_bin[idx_min]
        # Find maximum Y within bing range
        idx_max = np.argmax(Y[idx_bin])
        ub[i] = idx_bin[idx_max] 
    # Concatenate lowerbound and upperbound, and close points
    outer_contour = np.hstack((lb, ub[::-1], lb[0]))
    x_contour = X[outer_contour.astype(int)]
    y_contour = Y[outer_contour.astype(int)]
    return x_contour, y_contour



def PhasePlot(X, Y, Z, Z2 = None, plot_type = 'scatter', n_step = 10, x_ax_label=None, y_ax_label=None, z_ax_label = None, z_legend_label=None, z2_legend_label=None, returnFig = False):

    # Plot colors
    c1 = 'k'
    c2 = 'rebeccapurple'
    c3 = 'mediumseagreen'

    X_lim = [np.min((0.5*np.min(X), 2*np.min(X))), np.max((0.5*np.max(X), 2*np.max(X)))]
    Y_lim = [np.min((0.5*np.min(Y), 2*np.min(Y))), np.max((0.5*np.max(Y), 2*np.max(Y)))]
    Z_lim = [np.min((0.8*np.min(Z), 1.2*np.min(Z))), np.max((0.8*np.max(Z), 1.2*np.max(Z)))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_types = {'scatter':ax.scatter,
            'line':ax.plot}

    plot_types[plot_type](X[::n_step], Y[::n_step], Z[::n_step], label=z_legend_label, c=c2, zorder=3)
    
    if Z2 is not None:
        plot_types[plot_type](X[::n_step], Y[::n_step], Z2[::n_step], label=z2_legend_label, c=c3, zorder=4)

        plot_types[plot_type](X[::n_step], np.array(Z[::n_step]).reshape(-1), Y_lim[1], zdir='y', c=c2, alpha=0.75)
        plot_types[plot_type](X[::n_step], np.array(Z2[::n_step]).reshape(-1), Y_lim[1], zdir='y', c=c3, alpha=0.5)

        plot_types[plot_type](Y[::n_step], np.array(Z[::n_step]).reshape(-1), X_lim[0], zdir='x', c=c2, alpha=0.75)
        plot_types[plot_type](Y[::n_step], np.array(Z2[::n_step]).reshape(-1), X_lim[0], zdir='x', c=c3, alpha=0.5)        

        ax.legend(loc='upper left')

    ax.set_ylim(Y_lim)
    ax.set_xlim(X_lim)
    ax.set_zlim(Z_lim)
    ax.set_xlabel(x_ax_label, labelpad=15)
    ax.set_ylabel(y_ax_label, labelpad=15)
    ax.set_zlabel(z_ax_label, labelpad=15)

    if returnFig:
        return fig
    else:
        plt.show()
        return None



def ShiftPlot_Quiver(X, Y1, Y2, Yref, plotParams = None, n_step = 2, inset_axes = None, showQuiver = True, c1 = 'mediumseagreen', c2 = 'mediumorchid', c3 = 'dimgrey', returnFig = False):

    def getVar(var, ref):
        params = {}
        if var in ref.keys():
            params = ref[var]
        return params

    diffY = Y2 - Y1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    legend_loc = 'best'
    if 'legend_loc' in plotParams:
        legend_loc = plotParams['legend_loc']
    
    quiver_cmap = 'viridis'
    if 'cmap' in plotParams:
        quiver_cmap = plotParams['cmap']

    if inset_axes is not None:
        for axes in inset_axes:
            zoom_inset = getVar('inset_zoom', axes)
            axins = zoomed_inset_axes(ax, zoom_inset, bbox_transform=ax.transAxes, **getVar('inset_params', axes))
            idx_min = getVar('idx_min', axes)
            idx_max = getVar('idx_max', axes)
            y1_kwargs = getVar('Y1', plotParams).copy()
            y1_kwargs.pop('label', None)
            y2_kwargs = getVar('Y2', plotParams).copy()
            y2_kwargs.pop('label', None)
            y3_kwargs = getVar('Yref', plotParams).copy()
            y3_kwargs.pop('label', None)
            axins.plot(X[idx_min:idx_max:n_step], Y1[idx_min:idx_max:n_step], c = c1, zorder = 3, **y1_kwargs)
            axins.plot(X[idx_min:idx_max:n_step], Y2[idx_min:idx_max:n_step], c = c2, zorder = 1, **y2_kwargs)
            axins.plot(X[idx_min:idx_max:n_step], Yref[idx_min:idx_max:n_step], c = c3, zorder = 0, **y3_kwargs)
            if showQuiver:
                quiver_scale = getVar('quiver_scale', axes)
                quiver_kwargs = getVar('Quiver', plotParams).copy()
                quiver_kwargs.pop('scale', None)
                axins.quiver(X[idx_min:idx_max:n_step], Y1[idx_min:idx_max:n_step], X[idx_min:idx_max:n_step]*0, diffY[idx_min:idx_max:n_step], np.sin(diffY[idx_min:idx_max:n_step]), 
                                zorder = 2, scale = quiver_scale, cmap = quiver_cmap, **quiver_kwargs)
            plt.setp(axins.get_xticklabels(), visible=False)
            plt.setp(axins.get_yticklabels(), visible=False)
            axins.tick_params(which='both', axis='both', length=0)
            box, p1, p2 = mark_inset(ax, axins, getVar('mark_inset_1', axes), getVar('mark_inset_2', axes), zorder=4, linewidth=1.5)
            if 'inset_color' in axes:
                box.set_color(axes['inset_color'])
                plt.setp([p1, p2], color=axes['inset_color'], linewidth=0.5)
                plt.setp(list(axins.spines.values()), linewidth=1.5, color=axes['inset_color'])


    ax.plot(X[::n_step], Y1[::n_step], c = c1, zorder = 3, **getVar('Y1', plotParams))
    ax.plot(X[::n_step], Y2[::n_step], c = c2, zorder = 1, **getVar('Y2', plotParams))
    ax.plot(X[::n_step], Yref[::n_step], c = c3, zorder = 0, **getVar('Yref', plotParams))
    if showQuiver:
        ax.quiver(X[::n_step], Y1[::n_step], X[::n_step]*0, diffY[::n_step], np.sin(diffY[::n_step]), zorder = 2, cmap=quiver_cmap, **getVar('Quiver', plotParams))

    ax.set_xlabel(getVar('x_label', plotParams))
    ax.set_ylabel(getVar('y_label', plotParams))
    ax.legend(loc=legend_loc)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    
    if returnFig:
        return fig
    else:
        plt.show()
        return None



def ShiftPlot_Contour(X, Y1, Y2, Yref, plotParams = None, n_step = 2, inset_axes = None, c1 = 'mediumseagreen', c2 = 'mediumorchid', c3 = 'k', bin_num = 100, returnFig = False):

    def getVar(var, ref):
        params = {}
        if var in ref.keys():
            params = ref[var]
        return params

    diffY = Y2 - Y1

    X_C, Y1_C = _findOuterContour(X, Y1, res=bin_num)
    X_C, Y2_C = _findOuterContour(X, Y2, res=bin_num)
    X_C, Yref_C = _findOuterContour(X, Yref, res=bin_num)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    y1_kwargs = getVar('Y1', plotParams).copy()
    y1_kwargs.pop('label', None)
    y2_kwargs = getVar('Y2', plotParams).copy()
    y2_kwargs.pop('label', None)
    y3_kwargs = getVar('Yref', plotParams).copy()
    y3_kwargs.pop('label', None)

    legend_loc = 'best'
    if 'legend_loc' in plotParams:
        legend_loc = plotParams['legend_loc']

    if inset_axes is not None:
        for axes in inset_axes:
            zoom_inset = getVar('inset_zoom', axes)
            axins = zoomed_inset_axes(ax, zoom_inset, bbox_transform=ax.transAxes, **getVar('inset_params', axes))
            idx_min = getVar('idx_min', axes)
            idx_max = getVar('idx_max', axes)
            axins.plot(X[idx_min:idx_max:n_step], Y1[idx_min:idx_max:n_step], c = c1, zorder = 3, **y1_kwargs)
            axins.plot(X[idx_min:idx_max:n_step], Y2[idx_min:idx_max:n_step], c = c2, zorder = 1, **y2_kwargs)
            axins.plot(X[idx_min:idx_max:n_step], Yref[idx_min:idx_max:n_step], c = c3, zorder = 0, **y3_kwargs)
            axins.set_xlim([X[idx_min], X[idx_max]])
            plt.setp(axins.get_xticklabels(), visible=False)
            plt.setp(axins.get_yticklabels(), visible=False)
            axins.tick_params(which='both', axis='both', length=0)
            patch, pp1, pp2 = mark_inset(ax, axins, getVar('mark_inset_1', axes), getVar('mark_inset_2', axes), zorder=0, linewidth=0.5)
            if np.max(Yref[idx_min:idx_max:n_step]) < 0:
                pp1.loc1 = getVar('mark_inset_1', axes)
                pp1.loc2 = getVar('mark_inset_2', axes)
                pp2.loc1 = getVar('mark_inset_2', axes)
                pp2.loc2 = getVar('mark_inset_1', axes)


    y1_kwargs.pop('alpha', None)
    y2_kwargs.pop('alpha', None)
    y3_kwargs.pop('alpha', None)
    ax.fill(X_C, Y1_C, c = c1, zorder=3, alpha = 0.5, **y1_kwargs)
    ax.fill(X_C, Y2_C, c = c2, zorder=1, alpha = 0.4, **y2_kwargs)
    ax.fill(X_C, Yref_C, c = c3, zorder=0, alpha = 0.2, **y3_kwargs)

    ax.plot(X[::n_step], Y1[::n_step], c = c1, zorder = 3, **getVar('Y1', plotParams))
    ax.plot(X[::n_step], Y2[::n_step], c = c2, zorder = 1, **getVar('Y2', plotParams))
    ax.plot(X[::n_step], Yref[::n_step], c = c3, zorder = 0, **getVar('Yref', plotParams))

    ax.set_xlabel(getVar('x_label', plotParams))
    ax.set_ylabel(getVar('y_label', plotParams))
    ax.legend(loc=legend_loc)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    plt.show()

    if returnFig:
        return fig
    else:
        plt.show()
        return None



def livePlot3D(X, Y, *args, axisParams = None, plotParams=None, useBlit=False, showProjection=True, d_frame=10, interval = 1, trail = 50, trailDecay = None, showPlot=False, parentFig = None, subplot = 111):

    def updateLines3D(num, dataLines, lines, lines2):
        if trailDecay is None:
            lag = num
        else:
            lag = trailDecay
        if lines2 is not None:
            for line, line2, data in zip(lines, lines2, dataLines):
                line.set_data(data[0:2, np.max([0, num-lag]):num])
                line.set_3d_properties(data[2, np.max([0, num-lag]):num])
                line2.set_data(data[0:2, np.max((0, num-trail)):num])
                line2.set_3d_properties(data[2, np.max((0, num-trail)):num])
            lines = lines + lines2
        else:
            for line, data in zip(lines, dataLines):
                line.set_data(data[0:2, np.max([0, num-lag]):num])
                line.set_3d_properties(data[2, :num])
        return lines


    def updateLines3D_proj(num, dataLines, lines, lines2, proj1, proj2, ylim, xlim):
        if trailDecay is None:
            lag = num
        else:
            lag = trailDecay
        if lines2 is not None:
            for line, line2, data in zip(lines, lines2, dataLines):
                line.set_data(data[0:2, np.max([0, num-lag]):num])
                line.set_3d_properties(data[2, np.max([0, num-lag]):num])
                line2.set_data(data[0:2, np.max((0, num-trail)):num])
                line2.set_3d_properties(data[2, np.max((0, num-trail)):num])
            lines = lines + lines2
        else:
            for line, data in zip(lines, dataLines):
                line.set_data(data[0:2, np.max([0, num-lag]):num])
                line.set_3d_properties(data[2, np.max([0, num-lag]):num])

        for p_line1, p_line2, data in zip(proj1, proj2, dataLines):
            p_line1.set_xdata(data[0, np.max([0, num-lag]):num])
            p_line1.set_ydata(ylim[1])
            p_line1.set_3d_properties(data[2, np.max([0, num-lag]):num])

            p_line2.set_xdata(xlim[0])
            p_line2.set_ydata(data[1, np.max([0, num-lag]):num])
            p_line2.set_3d_properties(data[2, np.max([0, num-lag]):num])   

        lines = lines + proj1 + proj2
        return lines


    def makeData(X, Y, *args):
        dataLines = []
        XY = np.vstack((X, Y))
        for z in args:
            dat = np.vstack((XY, z))
            dataLines.append(dat)

        return dataLines



    # Check if args are given
    if len(args) > 0:
        if parentFig is None:
            fig = plt.figure()
            ax = fig.add_subplot(subplot, projection='3d')
        else:
            fig = parentFig
            currAx = fig.axes[-1]
            numRows = currAx.numRows
            numCols = currAx.numCols
            pos = (currAx.rowNum + 1)*(currAx.colNum + 1)
            # subplotRow = np.ceil(((pos+1)/numCols))
            # subplotCol = numCols - ((pos+1) % numCols)
            ax = fig.add_subplot(int('{}{}{}'.format(numRows, numCols, pos+1)), projection='3d')



        data = makeData(X, Y, *args)

        X_lim = [np.min((0.4*np.min(X), 2.5*np.min(X))), np.max((0.4*np.max(X), 2.5*np.max(X)))]
        Y_lim = [np.min((0.4*np.min(Y), 2.5*np.min(Y))), np.max((0.4*np.max(Y), 2.5*np.max(Y)))]
        Z = np.vstack(args)
        Z_lim = [np.min((0.8*np.min(Z), 1.2*np.min(Z))), np.max((0.8*np.max(Z), 1.2*np.max(Z)))]        

        # Modify plotParams to highlight the current state of the animation
        if plotParams is not None:
            plotParams2 = plotParams.copy()
            if 'alpha' in plotParams.keys():
                alphas = plotParams['alpha']
                new_alphas = [np.min((1, 2*i)) for i in alphas]
                plotParams2.update({'alpha':new_alphas})
            if 'label' in plotParams.keys():
                plotParams2.pop('label')
            if 'linestyle' not in plotParams.keys():
                plotParams.update({'linestyle':['dashed']*len(args)})
                plotParams2.update({'linestyle':['solid']*len(args)})

            lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], **{k:v[i] for k, v in plotParams.items()})[0] for i, dat in enumerate(data)]
            lines2 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], **{k:v[i] for k, v in plotParams2.items()})[0] for i, dat in enumerate(data)]

            if showProjection:
                if 'alpha' in plotParams2.keys():
                    plotParams2.update({'alpha':alphas})
                if 'zorder' in plotParams2.keys():
                    plotParams2.pop('zorder')
                proj1 = [ax.plot(dat[0, 0:1], dat[2, 0:1], Y_lim[1], zorder=0, zdir='y', **{k:v[i] for k, v in plotParams2.items()})[0] for i, dat in enumerate(data)]
                proj2 = [ax.plot(dat[1, 0:1], dat[2, 0:1], X_lim[0], zorder=0, zdir='x', **{k:v[i] for k, v in plotParams2.items()})[0] for i, dat in enumerate(data)]
        
        else:
            lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
            lines2 = None

            if showProjection:
                proj1 = [ax.plot(dat[0, 0:1], dat[2, 0:1], Y_lim[1], zorder=0, zdir='y')[0] for dat in data]
                proj2 = [ax.plot(dat[1, 0:1], dat[2, 0:1], X_lim[0], zorder=0, zdir='x')[0] for dat in data]


        ax.set_xlim(X_lim)
        ax.set_ylim(Y_lim)
        ax.set_zlim(Z_lim)

        if axisParams is not None:
            ax.set_xlabel(axisParams['x_label'], labelpad=15)
            ax.set_ylabel(axisParams['y_label'], labelpad=15)
            ax.set_zlabel(axisParams['z_label'], labelpad=15)
        
        if plotParams is not None:
            if 'color' in plotParams.keys() and 'labels' in plotParams.keys():
                handles, labels = ax.get_legend_handles_labels()
                colors = plotParams['color']
                for i, c in enumerate(colors):
                    handles[i] = mpatches.Patch(color=c, label=labels[i])
                ax.legend(handles=handles)

            else:
                if 'labels' in plotParams.keys():
                    ax.legend()

        frms = np.arange(len(X))

        if showProjection:
            line_ani = animation.FuncAnimation(fig, updateLines3D_proj, frms[::d_frame], fargs=(data, lines, lines2, proj1, proj2, Y_lim, X_lim), interval=interval, blit=useBlit)
        else:
            line_ani = animation.FuncAnimation(fig, updateLines3D, frms[::d_frame], fargs=(data, lines, lines2), interval=interval, blit=useBlit)
        
        if showPlot:
            plt.show()

    else:
        raise ValueError('Expected at least 3 arguments, but only 2 were given.')

    return line_ani



def plotModelWithPI(y_true, y_preds, y_pred_vars, confidence = 0.95, x = None, returnFig = False, labels = None, colors = None, ylabel = None, xlabel = None, y_true_color = 'k', subplot=111):
    if confidence >= 1:
        print('[ WARNING ] User specified a confidence interval >= 1. Defaulting to 0.99.')
        confidence = 0.99
    z_conf = norm.ppf((1+confidence)/2)
    if x is None:
        x = np.arange(len(y_true))
    # defaultColors = ['mediumseagreen', 'mediumorchid']
    # defaultColors = ['#ffbe3c', 'mediumaquamarine', '#008bb4']
    defaultColors = ['#008bb4', 'mediumaquamarine', '#ffbe3c', 'firebrick']
    if colors is None:
        colors = cycle(defaultColors)
    else:
        colors = cycle(colors)
    if labels is None:
        labels = np.arange(len(y_preds))
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(subplot)
    for y_pred, y_var, label in zip(y_preds, y_pred_vars, labels):
        c = next(colors)
        PI_lower = np.array(y_pred.reshape(-1) - z_conf*np.sqrt(y_var.reshape(-1)))
        PI_upper = np.array(y_pred.reshape(-1) + z_conf*np.sqrt(y_var.reshape(-1)))
        ax.fill_between(x, PI_lower, PI_upper, color = c, alpha = 0.35)
        ax.plot(x, y_pred, c = c, label=label)
    ax.plot(x, y_true, c=y_true_color, linestyle='--', label='Measurement', alpha = 0.6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    ax.tick_params(which='both', direction='in', labelsize=14)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend()
    if returnFig:
        return fig
    else:
        return None



def _plotOverTime_3Axis(time, x, y, z, colors = ('seagreen', 'indianred', 'gold'), parentFig = None):
    if parentFig is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
    else:
        fig = parentFig
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]

    ax1.plot(time, x, c=colors[0])
    ax1.tick_params(which='both', direction='in', labelsize=14)
    ax1.set_xticklabels([])
    
    ax2.plot(time, y, c=colors[1])
    ax2.tick_params(which='both', direction='in', labelsize=14)
    ax2.set_xticklabels([])

    ax3.plot(time, z, c=colors[2])
    ax3.tick_params(which='both', direction='in', labelsize=14)
    ax3.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)

    return fig



def plotTrajectoryTime(time, x, y, z, colors = ('seagreen', 'indianred', 'gold'), parentFig = None):
    fig = _plotOverTime_3Axis(time, x, y, z, colors = colors, parentFig = parentFig)
    labels = (r'$\mathbf{X} \quad [m]$', r'$\mathbf{Y} \quad [m]$', r'$\mathbf{Z} \quad [m]$')
    for i, ax in enumerate(fig.axes):
        ax.set_ylabel(labels[i], fontsize = 16)
    return fig
    


def plotVelocityTime(time, u, v, w, colors = ('seagreen', 'indianred', 'gold'), parentFig = None):
    fig = _plotOverTime_3Axis(time, u, v, w, colors = colors, parentFig = parentFig)
    labels = (r'$\mathbf{u} \quad [ms^{-1}]$', r'$\mathbf{v} \quad [ms^{-1}]$', r'$\mathbf{w} \quad [ms^{-1}]$')
    for i, ax in enumerate(fig.axes):
        ax.set_ylabel(labels[i], fontsize = 16)
    return fig



def plotAccelerationTime(time, ax, ay, az, colors = ('seagreen', 'indianred', 'gold'), parentFig = None):
    fig = _plotOverTime_3Axis(time, ax, ay, az, colors = colors, parentFig = parentFig)
    labels = (r'$\mathbf{a_{x}} \quad [ms^{-2}]$', r'$\mathbf{a_{y}} \quad [ms^{-2}]$', r'$\mathbf{a_{z}} \quad [ms^{-2}]$')
    for i, ax in enumerate(fig.axes):
        ax.set_ylabel(labels[i], fontsize = 16)
    return fig



def plotPosVelAccTime(time, pos, vel, acc, colors = ('seagreen', 'indianred', 'gold'), parentFig = None):
    fig = _plotOverTime_3Axis(time, pos, vel, acc, colors = colors, parentFig = parentFig)
    labels = (r'$\mathbf{Position} \quad [m]$', r'$\mathbf{Velocity} \quad [ms^{-1}]$', r'$\mathbf{Acceleration} \quad [ms^{-2}]$')
    for i, ax in enumerate(fig.axes):
        ax.set_ylabel(labels[i], fontsize = 16)
    return fig



def Trajectory3D(time, X, Y, Z, Gradient = False, ColorMap = 'RdYlGn_r', parentFig = None, returnFig = True, subplot = 111, n_skip = 1):
    # Make inputs into arrays, if not already
    time = np.array(time)[::n_skip]
    X = np.array(X)[::n_skip]
    Y = np.array(Y)[::n_skip]
    Z = np.array(Z)[::n_skip]
    if parentFig is None:
        fig = plt.figure()
        ax = fig.add_subplot(subplot, projection='3d')
    else:
        fig = parentFig
        currAx = fig.axes[-1]
        numRows = currAx.numRows
        numCols = currAx.numCols
        pos = (currAx.rowNum + 1)*(currAx.colNum + 1)
        ax = fig.add_subplot(int('{}{}{}'.format(numRows, numCols, pos+1)), projection='3d')
    # If we want to represent time as a color gradient on the trajectory
    if Gradient:
        points = np.array([X, Y, Z]).transpose().reshape(-1, 1, 3)
        # Map points to line segments of neighboring points (e.g. point i with point i + 1)
        segments = np.concatenate([points[:-1], points[1:]], axis = 1)
        # Make collection of line segments
        lc = Line3DCollection(segments, cmap=ColorMap)
        # Color segments based on time
        lc.set_array(time)
        cmap = cm.get_cmap(ColorMap)
        ax.add_collection3d(lc)
        # Indicate where the motion began and where it ended, since temporal information is not displayed 
        ax.scatter(X[0], Y[0], Z[0], c = [cmap(0)], label = 'Start')
        ax.scatter(X[-1], Y[-1], Z[-1], c = [cmap(time[-1])], label = 'End')
        # ax.set_xlim3d(np.min([1.1*np.min(X), 0.9*np.min(X)]), np.max([1.1*np.max(X), 0.9*np.max(X)]))
        # ax.set_ylim3d(np.min([1.1*np.min(Y), 0.9*np.min(Y)]), np.max([1.1*np.max(Y), 0.9*np.max(Y)]))
        # ax.set_zlim3d(np.min([1.1*np.min(Z), 0.9*np.min(Z)]), np.max([1.1*np.max(Z), 0.9*np.max(Z)]))
    else:
        # ax.plot3D(X, Y, Z, label = 'Trajectory (displacement)')
        ax.plot3D(X, Y, Z, color = 'royalblue')
        # Indicate where the motion began and where it ended, since temporal information is not displayed 
        ax.scatter(X[0], Y[0], Z[0], c = 'g', label = 'Start')
        ax.scatter(X[-1], Y[-1], Z[-1], c = 'r', label = 'End')
    ax.legend()
    ax.set_xlabel(r'$\mathbf{x} \quad [m]$')
    ax.set_ylabel(r'$\mathbf{y} \quad [m]$')
    ax.set_zlabel(r'$\mathbf{z} \quad [m]$')
    if returnFig:
        return fig
    else:
        return None



def plotExcitations(target, excitationIdxs, ProcessedData, segregatedIdxs = None):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    segments = np.where(np.array(excitationIdxs[target][1:]) - np.array(excitationIdxs[target][:-1]) > 1)[0]
    if len(segments):
        eIdxs = []
        s0 = 0
        for s in segments:
            eIdxs.append(excitationIdxs[target][s0:s])
            s0 = s + 1
    else:
        eIdxs = [excitationIdxs[target]]

    ax.plot(ProcessedData[target], color = 'gainsboro')
    for i in eIdxs:
        if len(i):
            ax.plot(i, ProcessedData[target][i], color = '#008bb4', linewidth = 2)
            addXVSPAN(ax, i[0], i[-1], color = '#008bb4', alpha = 0.2)

    ax.set_xlabel(r'$\mathbf{Sample}$', fontsize = 14)
    ax.set_ylabel(r'$\mathbf{' + target + r'}$', fontsize = 14)
    handles = []
    addLegendLine(handles, color = 'gainsboro', label = f'{target} (all)')
    addLegendLine(handles, color = '#008bb4', linewidth = 1, label = f'{target} (excitation)')
    # addLegendPatch(handles, color = '#008bb4', alpha = 0.2)
    if segregatedIdxs is not None:
        for sIdx in segregatedIdxs:
            addVLINE(ax, sIdx, -1000, 1000, color = 'k')
        addLegendLine(handles, label = 'End of flight', color = 'k')

    ax.legend(handles = handles)

    plt.tight_layout()

    return fig


def ValidationRMSEExplorer(Data, Model, TargetColumn, Predictions, ValidationIdxs, TrainingIdxs, TestIdxs):
    c1 = '#ffbe3c'
    c2 = '#e67d0a'
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=1, ncols=5)
    ax1 = fig.add_subplot(gs[:, :3])
    ax2 = fig.add_subplot(gs[:, 3:], sharey = ax1)

    TargetIdxs_sorted = np.argsort(Data[TargetColumn])
    idxsLower = np.arange(0, int(0.25*len(Data)), 1)
    idxsMid = np.arange(int(0.25*len(Data)), int(0.75*len(Data)), 1)
    idxsTop = np.arange(int(0.75*len(Data)), len(Data), 1)

    TrainIdxs_Lower = np.array(list(set(TargetIdxs_sorted[idxsLower]).intersection(set(TrainingIdxs))))
    TrainIdxs_Mid = np.array(list(set(TargetIdxs_sorted[idxsMid]).intersection(set(TrainingIdxs))))
    TrainIdxs_Top = np.array(list(set(TargetIdxs_sorted[idxsTop]).intersection(set(TrainingIdxs))))

    TestIdxs_Lower = np.array(list(set(TargetIdxs_sorted[idxsLower]).intersection(set(TestIdxs))))
    TestIdxs_Mid = np.array(list(set(TargetIdxs_sorted[idxsMid]).intersection(set(TestIdxs))))
    TestIdxs_Top = np.array(list(set(TargetIdxs_sorted[idxsTop]).intersection(set(TestIdxs))))
        
    # DRANGE = np.nanmax(Data[TargetColumn]) - np.nanmin(Data[TargetColumn])
    TrainRMSE_Lower = Model._RMSE(Data[TargetColumn].to_numpy()[TrainIdxs_Lower], Predictions.prediction[TrainIdxs_Lower])
    TrainRMSE_Mid = Model._RMSE(Data[TargetColumn].to_numpy()[TrainIdxs_Mid], Predictions.prediction[TrainIdxs_Mid])
    TrainRMSE_Top = Model._RMSE(Data[TargetColumn].to_numpy()[TrainIdxs_Top], Predictions.prediction[TrainIdxs_Top])

    TestRMSE_Lower = Model._RMSE(Data[TargetColumn].to_numpy()[TestIdxs_Lower], Predictions.prediction[TestIdxs_Lower])
    TestRMSE_Mid = Model._RMSE(Data[TargetColumn].to_numpy()[TestIdxs_Mid], Predictions.prediction[TestIdxs_Mid])
    TestRMSE_Top = Model._RMSE(Data[TargetColumn].to_numpy()[TestIdxs_Top], Predictions.prediction[TestIdxs_Top])

    if ValidationIdxs is not None:
        ValidIdxs_Lower = np.array(list(set(TargetIdxs_sorted[idxsLower]).intersection(set(ValidationIdxs))))
        ValidIdxs_Mid = np.array(list(set(TargetIdxs_sorted[idxsMid]).intersection(set(ValidationIdxs))))
        ValidIdxs_Top = np.array(list(set(TargetIdxs_sorted[idxsTop]).intersection(set(ValidationIdxs))))

        ValidRMSE_Lower = Model._RMSE(Data[TargetColumn].to_numpy()[ValidIdxs_Lower], Predictions.prediction[ValidIdxs_Lower])
        ValidRMSE_Mid = Model._RMSE(Data[TargetColumn].to_numpy()[ValidIdxs_Mid], Predictions.prediction[ValidIdxs_Mid])
        ValidRMSE_Top = Model._RMSE(Data[TargetColumn].to_numpy()[ValidIdxs_Top], Predictions.prediction[ValidIdxs_Top])
    else:
        ValidRMSE_Lower = np.nan
        ValidRMSE_Mid = np.nan
        ValidRMSE_Top = np.nan

    print('[ INFO ] Splitting RMSE by quartile:')
    print(f'[ INFO ]\t Upper 25% of {TargetColumn} measured range:')
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Training (X)', f'{np.around(TrainRMSE_Top, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Test (T)', f'{np.around(TestRMSE_Top, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Validation (V)', f'{np.around(ValidRMSE_Top, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Ratio (X/T)', f'{TrainRMSE_Top/TestRMSE_Top}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Ratio (V/T)', f'{ValidRMSE_Top/TestRMSE_Top}'))
    print(f'[ INFO ]\t Interquartile {TargetColumn} measured range:')
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Training (X)', f'{np.around(TrainRMSE_Mid, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Test (T)', f'{np.around(TestRMSE_Mid, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Validation (V)', f'{np.around(ValidRMSE_Mid, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Ratio (X/T)', f'{TrainRMSE_Mid/TestRMSE_Mid}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Ratio (V/T)', f'{ValidRMSE_Mid/TestRMSE_Mid}'))
    print(f'[ INFO ]\t Lower 25% of {TargetColumn} measured range:')
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Training (X)', f'{np.around(TrainRMSE_Lower, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Test (T)', f'{np.around(TestRMSE_Lower, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Validation (V)', f'{np.around(ValidRMSE_Lower, 6)}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Ratio (X/T)', f'{TrainRMSE_Lower/TestRMSE_Lower}'))
    print(f'[ INFO ]\t \t' + '{:<15}: {:<10}'.format('Ratio (V/T)', f'{ValidRMSE_Lower/TestRMSE_Lower}'))

    # TextTop = f'Upper RMSE (Training): {np.around(TrainRMSE_Top, 6)}\nUpper RMSE (Test): {np.around(TestRMSE_Top, 6)}\nUpper RMSE (Validation): {np.around(ValidRMSE_Top, 6)}'
    # TextMid = f'Middle RMSE (Training): {np.around(TrainRMSE_Mid, 6)}\nMiddle RMSE (Test): {np.around(TestRMSE_Mid, 6)}\nMiddle RMSE (Validation): {np.around(ValidRMSE_Mid, 6)}'
    # TextLower = f'Lower RMSE (Training): {np.around(TrainRMSE_Lower, 6)}\nLower RMSE (Test): {np.around(TestRMSE_Lower, 6)}\nLower RMSE (Validation): {np.around(ValidRMSE_Lower, 6)}'

    TextHeader = '{:<10} {:<10} {:<10} {:<10}'.format('RMSE', 'Upper', 'Middle', 'Lower')
    TextTrain = '{:<10} {:<10} {:<10} {:<10}'.format('Train.', np.around(TrainRMSE_Top, 6), np.around(TrainRMSE_Mid, 6), np.around(TrainRMSE_Lower, 6))
    TextTest = '{:<10} {:<10} {:<10} {:<10}'.format('Test. ', np.around(TestRMSE_Top, 6), np.around(TestRMSE_Mid, 6), np.around(TestRMSE_Lower, 6))
    TextValid = '{:<10} {:<10} {:<10} {:<10}'.format('Valid.', np.around(ValidRMSE_Top, 6), np.around(ValidRMSE_Mid, 6), np.around(ValidRMSE_Lower, 6))
    RMSETable = f'{TextHeader}\n{TextTrain}\n{TextTest}\n{TextValid}'

    font = {'family': 'serif',
        'color':  'k',
        'weight': 'bold',
        'size': 12,
        }

    samples = np.arange(0, len(Data), 1)
    ax1.plot(samples, Data[TargetColumn], color = 'gray', label = 'Measurement', alpha = 0.5)
    # ax1.text(100, Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsTop[0]]], TextTop, fontdict=font)
    # ax1.text(100, Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsMid[0]]], TextMid, fontdict=font)
    # ax1.text(100, Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsLower[0]]], TextLower, fontdict=font)
    ax1.text(0.05, 0.87, RMSETable, transform = ax1.transAxes, fontdict=font)
    ylims = ax1.get_ylim()
    xlims = ax1.get_xlim()
    ax1.plot(samples, Predictions.prediction, color = '#008bb4', label = 'Prediction', alpha = 0.5)
    ax1.fill_between([0-0.5*len(Data), 1.5*len(Data)], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsLower[0]]], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsLower[-1]]], color = c1, hatch = '/', alpha = 0.2)
    ax1.fill_between([0-0.5*len(Data), 1.5*len(Data)], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsMid[0]]], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsMid[-1]]], facecolor = 'whitesmoke', edgecolor = 'gainsboro', alpha = 0.7)
    ax1.fill_between([0-0.5*len(Data), 1.5*len(Data)], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsTop[0]]], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsTop[-1]]], color = c1, hatch = '\\', alpha = 0.2)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.set_xlabel(r'$\mathbf{Sample}$, -', fontsize = 14)
    ax1.set_ylabel(r'$\mathbf{' + TargetColumn + r'}$, -', fontsize = 14)
    handles, _ = ax1.get_legend_handles_labels()
    addLegendPatch(handles, color = c1, hatch = '\\', alpha = 0.2, label = 'Upper quartile (25%)')
    addLegendPatch(handles, facecolor = 'whitesmoke', edgecolor = 'gainsboro', alpha = 0.7, label = 'Middle Interquartile (50%)')
    addLegendPatch(handles, color = c1, hatch = '/', alpha = 0.2, label = 'Lower quartile (25%)')
    ax1.legend(handles=handles, loc='upper right')
    prettifyAxis(ax1)

    mesh = np.linspace(np.nanmin(Data[TargetColumn]), np.nanmax(Data[TargetColumn]), 1000)
    TrainingDataDist = stats.gaussian_kde(Data[TargetColumn].to_numpy()[TrainingIdxs])
    TestDataDist = stats.gaussian_kde(Data[TargetColumn].to_numpy()[TestIdxs])
    ValidationDataDist = stats.gaussian_kde(Data[TargetColumn].to_numpy()[ValidationIdxs])
    ax2.plot(TrainingDataDist(mesh), mesh, color = 'tab:blue')
    ax2.fill_betweenx(mesh, mesh*0, TrainingDataDist(mesh), color = 'tab:blue', alpha = 0.5)
    ax2.plot(TestDataDist(mesh), mesh, color = 'tab:orange')
    ax2.fill_betweenx(mesh, mesh*0, TestDataDist(mesh), color = 'tab:orange', alpha = 0.5)
    ax2.plot(ValidationDataDist(mesh), mesh, color = 'tab:green')
    ax2.fill_betweenx(mesh, mesh*0, ValidationDataDist(mesh), color = 'tab:green', alpha = 0.5)
    xlim = ax2.get_xlim()
    ax2.fill_between([0, len(Data)], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsLower[0]]], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsLower[-1]]], color = c1, hatch = '/', alpha = 0.2)
    ax2.fill_between([0, len(Data)], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsMid[0]]], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsMid[-1]]], facecolor = 'whitesmoke', edgecolor = 'gainsboro', alpha = 0.7)
    ax2.fill_between([0, len(Data)], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsTop[0]]], Data[TargetColumn].to_numpy()[TargetIdxs_sorted[idxsTop[-1]]], color = c1, hatch = '\\', alpha = 0.2)
    ax2.set_xlim(xlim)
    plt.setp(ax2.get_yticklabels(), visible = False)
    ax2.set_xlabel(r'$\mathbf{Density}$, -', fontsize = 14)
    handles = []
    addLegendPatch(handles, color = 'tab:blue', alpha = 0.5, label = 'Training data')
    addLegendPatch(handles, color = 'tab:orange', alpha = 0.5, label = 'Test data')
    addLegendPatch(handles, color = 'tab:green', alpha = 0.5, label = 'Validation data')
    addLegendPatch(handles, color = c1, hatch = '\\', alpha = 0.2, label = 'Upper quartile (25%)')
    addLegendPatch(handles, facecolor = 'whitesmoke', edgecolor = 'gainsboro', alpha = 0.7, label = 'Middle Interquartile (50%)')
    addLegendPatch(handles, color = c1, hatch = '/', alpha = 0.2, label = 'Lower quartile (25%)')
    ax2.legend(handles=handles, loc='upper right')
    prettifyAxis(ax2)

    plt.tight_layout()

    return fig