# Library imports
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from typing import Optional

# Local imports
from common import plotter
from processing import utility

myOrange = '#e67d0a'
myBlue = '#008bb4'
myGreen = 'mediumseagreen'
myYellow = '#ffbe3c'
myRed = 'firebrick'
myGrey = 'gainsboro'
myVelvet = 'mediumvioletred'
myOrangeRed = '#E5340B'
myPurple = 'mediumorchid'



def _area(poly: np.ndarray) -> float:
    x, y = poly[:,0], poly[:,1]
    return 0.5 * np.sum(x*np.roll(y,-1) - y*np.roll(x,-1))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if _area(poly) > 0 else poly[::-1].copy()


def _angle_deg(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    nu = float(np.hypot(u[0], u[1])) + eps
    nv = float(np.hypot(v[0], v[1])) + eps
    c = float(np.clip((u[0]*v[0] + u[1]*v[1])/(nu*nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _cross(ax, ay, bx, by):
    return ax*by - ay*bx


def find_closest_points(X, k, scale = True):
    '''
    For each point in X, finds the k closest points, also in X
    '''
    if scale:
        Xmin = np.nanmin(X, axis = 0)
        Xmax = np.nanmax(X, axis = 0)
        X = (X-Xmin)/(Xmax - Xmin)
    tree = KDTree(X, leaf_size=60, metric = 'euclidean')
    dist, idx = tree.query(X, k=k+1, return_distance=True)
    return dist, idx


def extract_iod_idxs(mdl, y, query_idxs, data, k = 100, metrics = {'PI':0.95, 'strict':False, 'ball':0.25}):
    '''
    Uses inputted indices to determine the intended operating domain (iod)

    If strict is true, then PI is over-written. 

    Idea:
    1. Find clusters of points
    2. Compute their inclusion metrics
        - RMSE 
        - PI estimate
        - Density?
    '''
    y_true = y[query_idxs]
    # Get prediction
    y, var = mdl.predict(data.loc[query_idxs, :])
    PIL, PIW = utility.buildIntervalBounds(metrics['PI'], y, var)
    in_out = np.zeros(len(y)) # 1 means in, 0 means out
    in_out[((PIW.__array__().reshape(-1) - y_true) > 0) & ((y_true - PIL.__array__().reshape(-1)) > 0)] = 1
    # Get regressors
    Regressors = mdl.TrainedModel['Model']['Regressors']
    hasBias = True if 'bias' in Regressors else False
    # Extract model matrix
    MAT = np.asarray(mdl._techniqueModule._BuildRegressorMatrix(Regressors, data.loc[query_idxs, :], hasBias = hasBias))*np.asarray(mdl.TrainedModel['Model']['Parameters']).reshape(-1)
    # Get cluster
    idxs_to_keep = set(np.arange(0, len(y)).tolist())
    doPICP = True
    for i, reg in enumerate(Regressors):
        if reg != 'bias':
            X = np.hstack([MAT[:, i].reshape(-1, 1), y]).__array__()
            dist, idxs = find_closest_points(X, k = k, scale = False)
            include = np.ones(len(idxs[:, 1:]))
            if 'strict' in metrics.keys():
                if metrics['strict']:
                    include[~in_out.astype(bool)] = 0
                    doPICP = False
            if 'PI' in metrics.keys() and doPICP:
                include[np.nanmean(in_out[idxs], axis = 1) < metrics['PI']] = 0
            if 'ball' in metrics.keys():
                furthest = dist[:, -1]
                include[furthest > metrics['ball']] = 0
            # #
            kdist_mask = np.zeros(len(include))
            kdist = dist[np.where(include), -1].reshape(-1)
            kdist_mask[np.where(include)] = kdist
            include[kdist_mask > np.sort(kdist)[int(len(kdist)*0.975)]] = 0
            # #
            k0dist_mask = np.zeros(len(include))
            k0dist = dist[np.where(include), 1].reshape(-1)
            k0dist_mask[np.where(include)] = k0dist
            include[k0dist_mask > np.sort(k0dist)[int(len(k0dist)*0.975)]] = 0 # Exclude points which are isolated. 
            idxs_to_keep = idxs_to_keep.intersection(set(np.where(include)[0].tolist()))
    #
    idxs_to_keep = list(idxs_to_keep)
    return idxs_to_keep


def convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    hull = ConvexHull(points)
    return _ensure_ccw(points[hull.vertices])


def make_concave(points: np.ndarray,
                proportion_threshold: float = 1e-5,
                passes: int = 1,
                min_t_ratio: float = 0.0,
                hull_vertices: Optional[np.ndarray] = None,
                *,
                smoothen: bool = True,
                tip_min_deg: float = 45.0,
                bisection_iters: int = 22,
                eps: float = 1e-12) -> np.ndarray:
    """
    Exact concave hull by counting true triangle membership.
    - For edge (v1,v2), apex X = C + u*t, u = unit(M-C), t in [0,L].
    - Count points inside TRIANGLE(v1,v2,X) with half-planes linear in t.
    - Bisection to find deepest t with clipped_fraction <= proportion_threshold.
    - If that requires depth > max_depth, we SKIP inserting on that edge (keeps convex edge).
    - Optional apex-angle gate (same as your earlier function).
    """
    P = np.ascontiguousarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be (N,2)")
    if not np.isfinite(P).all():
        raise ValueError("points contain NaN/Inf")
    N = len(P)
    if N == 0:
        raise ValueError("empty point set")
    poly = convex_hull_vertices(P) if hull_vertices is None else _ensure_ccw(
        np.ascontiguousarray(hull_vertices, dtype=np.float64)
    )
    C = P.mean(axis=0)
    k_allowed = max(0, int(np.floor(proportion_threshold * N + 1e-9)))  
    for _ in range(max(1, passes)):
        H = len(poly)
        new_vs = []
        for i in range(H):
            v1 = poly[i]
            v2 = poly[(i+1) % H]
            new_vs.append(v1)
            M = 0.5*(v1 + v2)
            d = M - C
            L = float(np.hypot(d[0], d[1]))
            if L < eps:
                continue
            u = d / L  
            Pv1 = P - v1
            Pv2 = P - v2
            PC  = P - C
            e12 = v2 - v1
            s0 = _cross(e12[0], e12[1], Pv1[:,0], Pv1[:,1])
            base_side = np.sign(_cross(e12[0], e12[1], (C - v1)[0], (C - v1)[1]))
            if base_side == 0:
                continue
            side_mask = (s0 >= -eps) if base_side > 0 else (s0 <= eps)
            if not np.any(side_mask):
                continue
            Pv2m = Pv2[side_mask]
            PCm  = PC[side_mask]
            s0m  = s0[side_mask]
            A1 = _cross((C - v2)[0], (C - v2)[1], Pv2m[:,0], Pv2m[:,1])
            B1 = _cross(u[0], u[1],                 Pv2m[:,0], Pv2m[:,1])
            A2       = _cross((v1 - C)[0], (v1 - C)[1], PCm[:,0], PCm[:,1])
            Bu_p     = _cross(u[0], u[1],                 PCm[:,0], PCm[:,1])
            const_u  = _cross((v1 - C)[0], (v1 - C)[1],   u[0],    u[1])
            B2_total = Bu_p + const_u

            def count_inside(t: float) -> int:
                s1t = A1 + t*B1
                s2t = A2 - t*B2_total
                pos = (s0m >= -eps) & (s1t >= -eps) & (s2t >= -eps)
                neg = (s0m <=  eps) & (s1t <=  eps) & (s2t <=  eps)
                return int(np.count_nonzero(pos | neg))
            
            t_hi = L * (1.0 - 1e-6)             
            t_lo = max(min_t_ratio * L, 0.0)    
            if count_inside(t_hi) > k_allowed:
                continue
            if count_inside(t_lo) <= k_allowed:
                t_star = t_lo
            else:
                lo, hi = t_lo, t_hi
                for _it in range(bisection_iters):
                    mid = 0.5*(lo + hi)
                    if count_inside(mid) <= k_allowed:
                        hi = mid
                    else:
                        lo = mid
                t_star = hi
            X = C + u * t_star
            if smoothen:
                tip = _angle_deg(v1 - X, v2 - X)
                if tip < tip_min_deg:
                    continue
            new_vs.append(X)
        poly = _ensure_ccw(np.vstack(new_vs))
    return poly


def estimate_iod(model, x, y, k = 200, PI_conf = 0.95, return_indices = False):
    # Assume x is DataFrame, assume Y is also dataFrame
    if not hasattr(model, 'TrainedModel'):
        raise ValueError('Incomplete model input. Expected trained SysID.Model object.')
    if PI_conf > 1 or PI_conf < 0:
        raise ValueError('Only 0 < PI_conf < 1 is allowed.')
    IOD = {}
    # Get the indices of the iod points
    idxs = x.index.to_numpy()
    # idxs_iod = extract_iod_idxs(model, y, idxs, x, k = k, metrics = {'PI':PI_conf})
    # idxs_iod = extract_iod_idxs(model, y, idxs, x, k = k, metrics = {'PI':PI_conf, 'strict':True})
    idxs_iod = extract_iod_idxs(model, y, idxs, x, k = k, metrics = {'PI':PI_conf, 'strict':True, 'ball':0.1})
    # Save to IOD dict
    IOD.update({'indices':{'iod_indices':idxs_iod, 'all_indices':idxs}})
    # Estimate a convexhull, per regressor
    Regressors = model.TrainedModel['Model']['Regressors']
    hasBias = True if 'bias' in Regressors else False
    params = np.asarray(model.TrainedModel['Model']['Parameters']).reshape(-1)
    MAT = np.asarray(model._techniqueModule._BuildRegressorMatrix(Regressors, x, hasBias = hasBias))*params
    cvx_hulls = {}
    ccv_hulls = {}
    for i, reg in enumerate(Regressors):
        # No point to check bias, as it is constant
        if reg != 'bias':
            _points = np.vstack([MAT[idxs_iod, i].reshape(-1), y[idxs_iod].to_numpy().reshape(-1)]).T
            _all_points = np.vstack([MAT[idxs, i].reshape(-1), y[idxs].to_numpy().reshape(-1)]).T
            _cvx_hull = convex_hull_vertices(_points)
            _cvx_ratio = ratio_in_hull(_all_points, _cvx_hull)
            cvx_hulls.update({reg:{'ratio':_cvx_ratio, 'hull':_cvx_hull.tolist()}})
            _ccv_hull = make_concave(_points, hull_vertices=_cvx_hull)
            _ccv_ratio = ratio_in_hull(_all_points, _ccv_hull)
            ccv_hulls.update({reg:{'ratio':_ccv_ratio, 'hull':_ccv_hull.tolist()}})
    IOD.update({'convex':cvx_hulls})
    IOD.update({'concave':ccv_hulls})
    if return_indices:
        return IOD, idxs_iod
    else:
        return IOD




def points_in_polygon(points: np.ndarray, poly: np.ndarray, include_boundary: bool = True, chunk_size: int = 200_000) -> np.ndarray:
    """
    - points: (N,2) float/ints
    - poly:   (M,2) polygon vertices (open or closed ring; order doesnt matter)
    Returns:  boolean mask of length N (True = inside, or on boundary if include_boundary=True)
    """
    P = np.asarray(points, dtype=np.float64)
    V = np.asarray(poly, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2: raise ValueError("points must be (N,2)")
    if V.ndim != 2 or V.shape[1] != 2: raise ValueError("poly must be (M,2)")
    if len(V) < 3: return np.zeros(len(P), dtype=bool)
    if not np.allclose(V[0], V[-1]):
        V = np.vstack([V, V[0]])
    x1, y1 = V[:-1, 0], V[:-1, 1]
    x2, y2 = V[ 1:, 0], V[ 1:, 1]
    dx, dy = x2 - x1, y2 - y1
    # tolerances
    scale = max(np.ptp(V[:,0]), np.ptp(V[:,1]), 1.0)
    eps = 1e-12 * scale
    N = len(P)
    inside = np.zeros(N, dtype=bool)
    # process in chunks to avoid N x M temporary arrays getting too large
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        x = P[start:stop, 0][:, None]  # (n,1)
        y = P[start:stop, 1][:, None]  # (n,1)
        y1_gt_y = y1[None, :] > y
        y2_gt_y = y2[None, :] > y
        straddles = y1_gt_y ^ y2_gt_y  # (n, m-1)
        x_inter = x1[None, :] + (y - y1[None, :]) * dx[None, :] / (dy[None, :] + (dy[None, :] == 0)*1.0)
        crosses = straddles & (x < x_inter)
        in_odd = np.count_nonzero(crosses, axis=1) % 2 == 1
        if include_boundary:
            cross = (x - x1[None, :]) * dy[None, :] - (y - y1[None, :]) * dx[None, :]
            on_line = np.abs(cross) <= eps
            minx = np.minimum(x1, x2)[None, :] - eps
            maxx = np.maximum(x1, x2)[None, :] + eps
            miny = np.minimum(y1, y2)[None, :] - eps
            maxy = np.maximum(y1, y2)[None, :] + eps
            in_box = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
            on_edge = np.any(on_line & in_box, axis=1)
            inside[start:stop] = in_odd | on_edge
        else:
            inside[start:stop] = in_odd
    return inside


def ratio_in_hull(points: np.ndarray, hull_vertices: np.ndarray, include_boundary: bool = True, chunk_size: int = 200_000) -> float:
    """
    Returns the fraction of points that lie inside (or on) the given concave hull polygon.
    """
    mask = points_in_polygon(points, hull_vertices,
                             include_boundary=include_boundary,
                             chunk_size=chunk_size)
    return float(np.mean(mask))


def plot_iod_2D(ax, iod_points, X, Y, color = myGreen, linewidth = 2, fill = False):
    points = np.vstack([X[iod_points], Y[iod_points]]).T
    hull = ConvexHull(points)
    contour = np.r_[hull.vertices, hull.vertices[0]]
    ax.plot(points[contour, 0], points[contour, 1], color = color, linewidth = linewidth, zorder = 1000)
    if fill:
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha = 0.2)


def lmapper(reg):
    lreg = reg.replace('(w2_1 + w2_2 + w2_3 + w2_4)', r'\sum_{i=1}^{4}\omega^{2}_{i}')
    lreg = lreg.replace(' ', '')
    lreg = lreg.replace('(', '{').replace(')', "}")
    lreg = lreg.replace('w_tot', r'\omega_{tot}')
    lreg = lreg.replace('.0', '')
    lreg = lreg.replace('*', r' \cdot ')
    return plotter.makeBoldLabel(lreg)