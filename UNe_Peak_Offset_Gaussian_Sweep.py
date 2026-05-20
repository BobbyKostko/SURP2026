from astropy.io import fits
from scipy.ndimage import median_filter
import numpy as np
from photutils.detection import DAOStarFinder
from scipy.optimize import curve_fit, linear_sum_assignment
import matplotlib.pyplot as plt
import math
import pandas as pd
import re
import matplotlib.colors as mcolors
import colorsys
import os


def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    xo, yo = float(xo), float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def process_fits(filename, NN, fwhm, threshold, hot_mask):
    # Load FITS image
    print(filename)
    data = fits.getdata(filename)
    if hot_mask is not None:
        hot_mask = fits.getdata(hot_mask)
        data = data * hot_mask
    data[data < 0] = 0

    # Detect sources
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*np.std(data))
    sources = daofind(data - np.median(data))

    #print(sources)
    results = []
    ny, nx = data.shape

    for src in sources:
        x, y = src['xcentroid'], src['ycentroid']

        # Define sub-image boundaries
        x0 = int(x - NN//2)
        x1 = int(x + NN//2)
        y0 = int(y - NN//2)
        y1 = int(y + NN//2)

        if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
            continue

        sub_image = data[y0:y1, x0:x1]

        # Coordinate grid
        X, Y = np.meshgrid(np.arange(sub_image.shape[1]), np.arange(sub_image.shape[0]))

        # Initial guess
        initial_guess = (sub_image.max(), NN//2, NN//2, 2.5, 2.5, 0, np.median(sub_image))

        try:
            popt, pcov = curve_fit(gaussian_2d,
                                   (X, Y),
                                   sub_image.ravel(),
                                   p0=initial_guess)

            # popt = (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
            # xo, yo are in sub-image pixel coordinates; convert to full-image coordinates.
            xo_sub, yo_sub = float(popt[1]), float(popt[2])
            x_fit = float(x0 + xo_sub)
            y_fit = float(y0 + yo_sub)

            # 1-sigma uncertainties from covariance (may be inf/nan if fit is ill-conditioned).
            x_fit_err = float(np.sqrt(pcov[1, 1])) if pcov is not None and pcov.shape[0] > 2 else float("nan")
            y_fit_err = float(np.sqrt(pcov[2, 2])) if pcov is not None and pcov.shape[0] > 2 else float("nan")

            results.append({
                'x_peak': x,
                'y_peak': y,
                'x_fit': x_fit,
                'y_fit': y_fit,
                'x_fit_err': x_fit_err,
                'y_fit_err': y_fit_err,
                'fit_params': popt,
                'fit_pcov': pcov,
                'sub_image': sub_image
            })

        except RuntimeError:
            continue

    return data, results, sources


def filter_results_by_fwhm(
    results,
    fwhm_x_min,
    fwhm_x_max,
    fwhm_y_min,
    fwhm_y_max,
):
    """
    Filter fitted peaks by Gaussian FWHM in x and y.

    Only keep peaks whose fitted Gaussian FWHM is within the provided x and y limits.
    1.5-9.0 and 0.0-5.0 is consistent with testing
    """
    filtered = []
    for r in results:
        # popt = (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
        _, _, _, sigma_x, sigma_y, _, _ = r["fit_params"]
        fwhm_x = 2.355 * abs(sigma_x)
        fwhm_y = 2.355 * abs(sigma_y)

        if (fwhm_x_min <= fwhm_x <= fwhm_x_max) and (fwhm_y_min <= fwhm_y <= fwhm_y_max):
            filtered.append(r)

    return filtered


def results_to_xy(results):
    if not results:
        return np.empty((0, 2), dtype=float)
    # Prefer fitted Gaussian centers when available; fall back to DAOStarFinder centroids.
    if "x_fit" in results[0] and "y_fit" in results[0]:
        return np.array([(r.get("x_fit", r["x_peak"]), r.get("y_fit", r["y_peak"])) for r in results], dtype=float)
    return np.array([(r["x_peak"], r["y_peak"]) for r in results], dtype=float)


def inverse_variance_weighted_mean(values, errors):
    v = np.asarray(values, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    if not np.any(ok):
        return float("nan"), float("nan")
    w = 1.0 / (e[ok] ** 2)
    mu = float(np.sum(w * v[ok]) / np.sum(w))
    mu_err = float(1.0 / np.sqrt(np.sum(w)))
    return mu, mu_err


def pairwise_peak_distances(xy_a, xy_b):
    """
    Compute Euclidean distance magnitude between all points in `xy_a` and `xy_b`.

    Returns
    -------
    dists : ndarray, shape (N, M)
        dists[i, j] = distance between xy_a[i] and xy_b[j]
    """
    xy_a = np.asarray(xy_a, dtype=float)
    xy_b = np.asarray(xy_b, dtype=float)

    if xy_a.size == 0 or xy_b.size == 0:
        return np.empty((xy_a.shape[0] if xy_a.ndim == 2 else 0,
                         xy_b.shape[0] if xy_b.ndim == 2 else 0), dtype=float)

    if xy_a.ndim != 2 or xy_a.shape[1] != 2:
        raise ValueError(f"xy_a must be shape (N, 2), got {xy_a.shape}")
    if xy_b.ndim != 2 or xy_b.shape[1] != 2:
        raise ValueError(f"xy_b must be shape (M, 2), got {xy_b.shape}")

    diffs = xy_a[:, None, :] - xy_b[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def compare_peaks_between_images(results_a, results_b, max_distance=None):
    """
    Compare peak positions between two `process_fits` runs.

    Parameters
    ----------
    results_a, results_b : list[dict]
        The `results` objects returned by `process_fits` (index 1 of the return tuple).
    max_distance : float or None
        If set, filter nearest-neighbor matches to those with distance <= max_distance.

    Returns
    -------
    out : dict
        Keys:
          - xy_a, xy_b: (N,2) and (M,2) arrays
          - distances: (N,M) matrix (Euclidean)
          - nearest_b_index_for_a: (N,) int array (argmin over columns)
          - nearest_b_distance_for_a: (N,) float array
          - nearest_a_index_for_b: (M,) int array (argmin over rows)
          - nearest_a_distance_for_b: (M,) float array
          - matches_a_to_b_within_max: list of (i_a, i_b, dist) tuples (only if max_distance provided)
    """
    xy_a = results_to_xy(results_a)
    xy_b = results_to_xy(results_b)

    dists = pairwise_peak_distances(xy_a, xy_b)

    if dists.size == 0:
        return {
            "xy_a": xy_a,
            "xy_b": xy_b,
            "distances": dists,
            "nearest_b_index_for_a": np.empty((xy_a.shape[0],), dtype=int),
            "nearest_b_distance_for_a": np.empty((xy_a.shape[0],), dtype=float),
            "nearest_a_index_for_b": np.empty((xy_b.shape[0],), dtype=int),
            "nearest_a_distance_for_b": np.empty((xy_b.shape[0],), dtype=float),
            "matches_a_to_b_within_max": [] if max_distance is not None else None,
        }

    nearest_b_idx = np.argmin(dists, axis=1)
    nearest_b_dist = dists[np.arange(dists.shape[0]), nearest_b_idx]

    nearest_a_idx = np.argmin(dists, axis=0)
    nearest_a_dist = dists[nearest_a_idx, np.arange(dists.shape[1])]

    matches_within = None
    if max_distance is not None:
        keep = nearest_b_dist <= float(max_distance)
        ia = np.where(keep)[0]
        matches_within = [(int(i), int(nearest_b_idx[i]), float(nearest_b_dist[i])) for i in ia]

    return {
        "xy_a": xy_a,
        "xy_b": xy_b,
        "distances": dists,
        "nearest_b_index_for_a": nearest_b_idx,
        "nearest_b_distance_for_a": nearest_b_dist,
        "nearest_a_index_for_b": nearest_a_idx,
        "nearest_a_distance_for_b": nearest_a_dist,
        "matches_a_to_b_within_max": matches_within,
    }


def match_peaks_global_optimal(results_a, results_b, max_distance=None):
    """
    One-to-one global optimal matching of peaks between two `process_fits` runs. Minimizes total distance between all matched peaks.

    Parameters
    ----------
    results_a, results_b : list[dict]
        The `results` objects returned by `process_fits` (index 1 of the return tuple).
    max_distance : float or None
        If set, only pairs with distance <= max_distance are counted as matches.
        Main program uses max_distance = 5 * median distance of all nearest peaks between two images 
             found using compare_peaks_between_images

    Returns
    -------
    out : dict
        Keys:
          - xy_a, xy_b: (N,2) and (M,2) arrays
          - distances: (N,M) matrix (Euclidean)
          - row_ind, col_ind: optimal assignment indices (Hungarian output)
          - assignment_distances: distances[row_ind, col_ind]
          - matches_within_max: list of (i_a, i_b, dist) tuples (None if max_distance is None)
          - unmatched_a_indices: 1D array of indices in A with no accepted match
          - unmatched_b_indices: 1D array of indices in B with no accepted match
    """
    xy_a = results_to_xy(results_a)
    xy_b = results_to_xy(results_b)

    dists = pairwise_peak_distances(xy_a, xy_b)

    # Handle empty cases cleanly
    if dists.size == 0:
        return {
            "xy_a": xy_a,
            "xy_b": xy_b,
            "distances": dists,
            "row_ind": np.array([], dtype=int),
            "col_ind": np.array([], dtype=int),
            "assignment_distances": np.array([], dtype=float),
            "matches_within_max": [] if max_distance is not None else None,
            "unmatched_a_indices": np.arange(xy_a.shape[0], dtype=int),
            "unmatched_b_indices": np.arange(xy_b.shape[0], dtype=int),
        }

    # Hungarian assignment on the full distance matrix
    row_ind, col_ind = linear_sum_assignment(dists)
    assign_dists = dists[row_ind, col_ind]

    # By default, treat all assignments as matches
    if max_distance is None:
        matched_a = row_ind
        matched_b = col_ind
        matches_within_max = None
    else:
        # Apply max-distance cutoff
        keep = assign_dists <= float(max_distance)
        matched_a = row_ind[keep]
        matched_b = col_ind[keep]
        matches_within_max = [
            (int(i_a), int(i_b), float(d))
            for i_a, i_b, d in zip(matched_a, matched_b, assign_dists[keep])
        ]

    # Anything not in matched_a / matched_b is considered unmatched
    all_a = np.arange(xy_a.shape[0], dtype=int)
    all_b = np.arange(xy_b.shape[0], dtype=int)
    unmatched_a = np.setdiff1d(all_a, matched_a)
    unmatched_b = np.setdiff1d(all_b, matched_b)

    return {
        "xy_a": xy_a,
        "xy_b": xy_b,
        "distances": dists,
        "row_ind": row_ind,
        "col_ind": col_ind,
        "assignment_distances": assign_dists,
        "matches_within_max": matches_within_max,
        "unmatched_a_indices": unmatched_a,
        "unmatched_b_indices": unmatched_b,
    }


def extract_match_offsets(matching):
    """
    Pulls specific (dx,dy) offsets for the matched peaks.

    Returns
    -------
    i_a : ndarray, shape (K,)
        Indices in the first image's results list.
    i_b : ndarray, shape (K,)
        Indices in the second image's results list.
    dx : ndarray, shape (K,)
        Δx = x_b - x_a for each accepted match.
    dy : ndarray, shape (K,)
        Δy = y_b - y_a for each accepted match.
    dist : ndarray, shape (K,)
        Euclidean distance for each accepted match.
    """
    xy_a = matching["xy_a"]
    xy_b = matching["xy_b"]

    if matching.get("matches_within_max") is not None:
        triples = matching["matches_within_max"]
        if not triples:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        i_a = np.array([t[0] for t in triples], dtype=int)
        i_b = np.array([t[1] for t in triples], dtype=int)
        dist = np.array([t[2] for t in triples], dtype=float)
    else:
        i_a = matching["row_ind"]
        i_b = matching["col_ind"]
        dist = matching["assignment_distances"]

    dx = xy_b[i_b, 0] - xy_a[i_a, 0]
    dy = xy_b[i_b, 1] - xy_a[i_a, 1]

    return i_a, i_b, dx, dy, dist


def compute_weighted_avg_offsets(results_ref, results_cmp, max_distance=None):
    """
    Compare peaks in `results_cmp` vs `results_ref` and compute inverse-variance weighted
    average dx/dy (cmp - ref) plus uncertainties, using centroid fit errors when available.
    """
    comparison = compare_peaks_between_images(results_ref, results_cmp)
    nearest_dists = comparison["nearest_b_distance_for_a"]
    if nearest_dists.size == 0:
        return {
            "dx_mean": float("nan"),
            "dx_err": float("nan"),
            "dy_mean": float("nan"),
            "dy_err": float("nan"),
            "n_matches": 0,
            "median_nearest": float("nan"),
            "max_distance_used": float("nan") if max_distance is None else float(max_distance),
        }

    median_nearest = float(np.median(nearest_dists))
    matching = match_peaks_global_optimal(results_ref, results_cmp, max_distance=max_distance)
    i_ref, i_cmp, dx, dy, _dist = extract_match_offsets(matching)

    if dx.size == 0:
        return {
            "dx_mean": float("nan"),
            "dx_err": float("nan"),
            "dy_mean": float("nan"),
            "dy_err": float("nan"),
            "n_matches": 0,
            "median_nearest": median_nearest,
            "max_distance_used": float("nan") if max_distance is None else float(max_distance),
        }

    # σ(dx) = sqrt(σ(x_cmp)^2 + σ(x_ref)^2), likewise for dy
    xerr_ref = np.array([results_ref[i].get("x_fit_err", np.nan) for i in i_ref], dtype=float)
    yerr_ref = np.array([results_ref[i].get("y_fit_err", np.nan) for i in i_ref], dtype=float)
    xerr_cmp = np.array([results_cmp[i].get("x_fit_err", np.nan) for i in i_cmp], dtype=float)
    yerr_cmp = np.array([results_cmp[i].get("y_fit_err", np.nan) for i in i_cmp], dtype=float)

    dx_err = np.sqrt(xerr_ref**2 + xerr_cmp**2)
    dy_err = np.sqrt(yerr_ref**2 + yerr_cmp**2)

    dx_mean, dx_mean_err = inverse_variance_weighted_mean(dx, dx_err)
    dy_mean, dy_mean_err = inverse_variance_weighted_mean(dy, dy_err)

    # Fallback to unweighted mean if uncertainties are missing/invalid.
    if not (np.isfinite(dx_mean) and np.isfinite(dx_mean_err)):
        dx_mean = float(np.mean(dx))
        dx_mean_err = float("nan")
    if not (np.isfinite(dy_mean) and np.isfinite(dy_mean_err)):
        dy_mean = float(np.mean(dy))
        dy_mean_err = float("nan")

    return {
        "dx_mean": float(dx_mean),
        "dx_err": float(dx_mean_err),
        "dy_mean": float(dy_mean),
        "dy_err": float(dy_mean_err),
        "n_matches": int(dx.size),
        "median_nearest": median_nearest,
        "max_distance_used": float("nan") if max_distance is None else float(max_distance),
    }


def main():
    """
    ============================
    Manual inputs (edit these)
    ============================
    """
    reference_filename = r"Y:\2D\20260128\iLocater_lab_20260128_0013_hxrgproc.fits"

# List of files to be compared
    compare_filenames = [
        r"Y:\2D\20260129\iLocater_lab_20260129_0013_hxrgproc.fits",
        r"Y:\2D\20260202\iLocater_lab_20260202_0013_hxrgproc.fits",
        r"Y:\2D\20260130\iLocater_lab_20260130_0063_hxrgproc.fits",
        r"Y:\2D\20260203\iLocater_lab_20260203_0009_hxrgproc.fits",
        r"Y:\2D\20260204\iLocater_lab_20260204_0008_hxrgproc.fits",
        r"Y:\2D\20260205\iLocater_lab_20260205_0013_hxrgproc.fits",
        r"Y:\2D\20260206\iLocater_lab_20260206_0014_hxrgproc.fits",
        r"Y:\2D\20260209\iLocater_lab_20260209_0013_hxrgproc.fits",
        r"Y:\2D\20260210\iLocater_lab_20260210_0013_hxrgproc.fits",
    ]

    hot_mask = r"hot_pixel_mask.fits" # can be None, but I wouldn't promote that

    # Peak-finding parameters. 0.3 threshold is good for 200-300 peaks per fibre. Lower it to get more peaks.
    NN = 12
    fwhm = 2.5
    threshold = 0.3

    # FWHM filtering on fitted Gaussians. Rough window to quickly get rid of bad fits.
    fwhm_x_min = 1.5
    fwhm_x_max = 9.0
    fwhm_y_min = 0.0
    fwhm_y_max = 5.0


    # Matching cutoff
    # If max_distance is None, uses max_distance_scale * median-nearest distance.
    max_distance = None
    max_distance_scale = 5.0

    # Output CSV (saved in this script's folder unless you use a full path)
    output_csv = r"UNe_peak_offset_summary.csv"

    ref_path = reference_filename
    cmp_paths = list(compare_filenames)

    # Find peaks for reference image
    _ref_data, ref_results, ref_sources = process_fits(
        ref_path,
        NN=NN,
        fwhm=fwhm,
        threshold=threshold,
        hot_mask=hot_mask,
    )
    ref_results = filter_results_by_fwhm(
        ref_results,
        fwhm_x_min=fwhm_x_min,
        fwhm_x_max=fwhm_x_max,
        fwhm_y_min=fwhm_y_min,
        fwhm_y_max=fwhm_y_max,
    )
    print(f"Reference: {ref_path}")
    print(f"Reference: found {len(ref_results)} peaks after FWHM filtering (raw sources: {len(ref_sources)})")

#Find peaks for each of the comparison images
    rows = []
    for cmp_path in cmp_paths:
        _cmp_data, cmp_results, cmp_sources = process_fits(
            cmp_path,
            NN=NN,
            fwhm=fwhm,
            threshold=threshold,
            hot_mask=hot_mask,
        )
        cmp_results = filter_results_by_fwhm(
            cmp_results,
            fwhm_x_min=fwhm_x_min,
            fwhm_x_max=fwhm_x_max,
            fwhm_y_min=fwhm_y_min,
            fwhm_y_max=fwhm_y_max,
        )

        # Match peaks between reference and comparison images.
        max_distance_used = max_distance
        if max_distance_used is None:
            comp_preview = compare_peaks_between_images(ref_results, cmp_results)
            nd = comp_preview["nearest_b_distance_for_a"]
            if nd.size > 0:
                max_distance_used = float(max_distance_scale * np.median(nd))
            else:
                max_distance_used = float("nan")

        # Computes dx, dy, and errors as a dictionary for each file, and throws them all in a list.
        stats = compute_weighted_avg_offsets(ref_results, cmp_results, max_distance=max_distance_used)
        rows.append(
            {
                "filename": os.path.basename(cmp_path),
                "path": cmp_path,
                "dx": stats["dx_mean"],
                "dx err": stats["dx_err"],
                "dy": stats["dy_mean"],
                "dy err": stats["dy_err"],
                "# of peaks raw": 0 if cmp_sources is None else len(cmp_sources),
                "# of peaks filtered": len(cmp_results),
                "# of matches": stats["n_matches"],
            }
        )

    # Makes a table from the list above.
    df = pd.DataFrame(rows)
    # Put the requested columns first.
    front = ["filename", "dx", "dx err", "dy", "dy err"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print("\nSummary (each file vs reference):")
        print(df.to_string(index=False))

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved CSV to {output_csv}")


if __name__ == "__main__":
    main()