import numpy as np
import math
import os
import argparse
from functools import lru_cache

# Shape parameters; all are required in the parameter file
PARAM_NAMES = [
    "L", "beam_rel", "F_over_L", "F_bow_over_L",
    "Cm", "Cp", "LCB", "bow_fullness", "L_pmb",
    "bow_convexity", "entrance_angle",
    "transom_imm", "transom_beam", "c_transom", "transom_rake",
    "keel_rise_short", "bilge_softness", "bilge_softness_aft",
    "flare_over_L", "flare_aft_over_L",
    "stem_radius_over_L", "stem_rake", "stem_curvature",
    "C_BB", "C_LPR", "C_ZB", "bulb_type", "bulb_fair_over_L", "bulb_rise", "bulb_implicit"
]

# Fixed hull data (not shape parameters): required in the parameter file together with
# PARAM_NAMES, but not part of the vector of shape parameters
FIXED_NAMES = ["volume"]
ALL_NAMES = PARAM_NAMES + FIXED_NAMES

# Internal geometric constants
RHO_SEA = 1.025              # sea water density [t/m^3]
MIN_CELL_FRAC = 0.25        # minimum cell in v as a fraction of the mean section cell
BULB_STRETCH_FRAC = 0.25     # forward stretch (fraction of L) over which the bulb waterlines are extended
BULB_MIN_AMP = 0.30          # minimum bulb amplitude / B_B / 2 (bulb always visible even if the hull at the FP is already wide)
BULB_TOP_MAX = 0.95          # bulb top H_B at most at this fraction of the draft (bulb always submerged)
BULB_ZM_RANGE = (0.35, 0.70)  # range of Z_B / H_B spanned by bulb_type in [0, 1] (Delta -> Nabla)
TRANSOM_BLEND_FRAC = 0.08    # aft stretch (fraction of L) over which the transom rake is blended
BULB_FILL_TOP = 0.95         # aft of the FP the bulb top rises towards this fraction of the draft (fills the neck)
BULB_NODE_FRAC = 0.15        # share of the longitudinal grid nodes reserved for the bulb and its fairing
STEM_ROUND_LEN = 0.05        # length scale (fraction of L) of the stem rounding term; it fades out between 2 and 5 times this
Q_BOW = 2.0                  # bilge exponent at the stem: forward sections (V-shaped) always turn softly
STEM_FADE_DEPTH = 0.5        # below the waterline the stem rake fades out over this fraction of the draft
CP_FWD_RANGE = (0.55, 0.90)  # admissible forebody prismatic (below: knife-edge bow, always hollow)
CP_AFT_RANGE = (0.60, 0.95)  # admissible afterbody prismatic (transom included)
XM_RANGE = (0.40, 0.62)      # admissible position of the midship section (centre of the parallel middle body) / L
TOPSIDE_BLEND = 0.35         # fraction of the freeboard over which the section blends into the vertical side
C_BOW_RANGE = (0.52, 0.75)   # section coefficient at the forward end (limits of the section family)
CONVEX_MIN_P = 0.50          # minimum prismatic of a convex branch (below it the branch is necessarily S-shaped)
CONVEX_RAMP = 0.02           # above CONVEX_MIN_P the convex weight grows from 0 to its full value over this interval
SECTION_Q_GRID = np.arange(1.5, 14.01, 0.5)   # tabulated bilge exponents for the section family


LB_RANGE = (3.5, 6.0)        # admissible L / B (B maximum waterline beam)
BT_RANGE = (2.4, 3.6)        # admissible B / T


def beam_window(p, bt):
    """
    Interval of admissible beams B for the product B T = bt: L / B in LB_RANGE and
    B / T = B^2 / bt in BT_RANGE. Returns (B_min, B_max, empty); if the intersection is empty
    (volume incompatible with L, Cp and Cm) only the B / T interval is used and empty = True.
    """
    L = p['L']
    lo = max(L / LB_RANGE[1], math.sqrt(BT_RANGE[0] * bt))
    hi = min(L / LB_RANGE[0], math.sqrt(BT_RANGE[1] * bt))
    if lo <= hi:
        return lo, hi, False
    return math.sqrt(BT_RANGE[0] * bt), math.sqrt(BT_RANGE[1] * bt), True


def dimensions(p):
    """
    Absolute dimensions [m]. The displaced volume (complete hull: bulb and stem included)
    is given: B T = kappa volume / (Cp Cm L), with kappa found by iteration (_bt_correction) so
    that the volume of the complete hull is exact (kappa = 1 with the compensated bulb, except in
    extreme corners). B is chosen by beam_rel within beam_window (realistic L / B and
    B / T), T = B T / B (draft of the horizontal keel), H_deck = T + F_over_L L (deck at the
    transom). Returns (B_half, T, H_deck).
    """
    L = p['L']
    kappa = p['_bt_corr'] if '_bt_corr' in p else _bt_correction(p)
    bt = kappa * p['volume'] / (p['Cp'] * p['Cm'] * L)
    lo, hi, _ = beam_window(p, bt)
    b = lo + p['beam_rel'] * (hi - lo)
    t = bt / b
    return 0.5 * b, t, t + p['F_over_L'] * L


_BT_CACHE = {}


def _bt_correction(p):
    """
    Factor kappa on the product B T so that the complete hull has the given volume, corrected by
    fixed-point iteration (volume ~ kappa). With the implicit bulb (bulb_implicit = 1) kappa = 1
    already holds unless the compensated bare hull is not reachable (extreme corners, see
    _sac_layout); with an additive bulb, bulb and stem add volume.
    """
    key = tuple(float(p[n]) for n in ALL_NAMES)
    if key in _BT_CACHE:
        return _BT_CACHE[key]
    kappa = 1.0
    for _ in range(8):
        pk = dict(p, _bt_corr=kappa)
        vol = _volume_moment(pk, form_data(pk))[0]
        k_new = kappa * p['volume'] / vol
        done = abs(k_new - kappa) < 1e-6
        kappa = k_new
        if done:
            break
    if len(_BT_CACHE) > 64:
        _BT_CACHE.clear()
    _BT_CACHE[key] = kappa
    return kappa


# ---------------------------------------------------------------------------
# Sectional area curve (SAC)
# Each branch (from the midship section towards one end) is f(t), t in [0, 1], with
# f(0) = 1, f'(0) = f''(0) = 0 (C2 join with the parallel middle body or the other branch),
# f decreasing, given mean P. Two families:
# - "S-shaped" (hollow): one inflection, zero slope at the end if P < 3/4 (concave
#   waterlines near the stem, blade-like entrance);
# - convex: no inflection, finite slope at the end (convex waterlines).
# The branch used is the linear blend of the two with weight lambda (same mean P).
# ---------------------------------------------------------------------------

_M2 = 1.0 - 2.0 / 4.0 + 1.0 / 7.0      # mean of (1 - t^3)^2


def _mean_pow3(m):
    """Mean of (1 - t^3)^m over [0, 1] = Gamma(4/3) Gamma(m+1) / Gamma(m+4/3)."""
    return math.exp(math.lgamma(4.0 / 3.0) + math.lgamma(m + 1.0) - math.lgamma(m + 4.0 / 3.0))


@lru_cache(maxsize=4096)
def _hollow_coeffs(P):
    """
    S-shaped branch with mean P, finite curvature at the end:
    P >= 3/4: 1 - t^n, n = P / (1 - P) (actually convex);
    _M2 <= P < 3/4: blend al (1 - t^3)^2 + (1 - al)(1 - t^3);
    P < _M2: (1 - t^3)^m with m >= 2.
    """
    if P >= 0.75:
        return ('pow', P / (1.0 - P))
    if P >= _M2:
        return ('mix', (0.75 - P) / (0.75 - _M2))
    lo, hi = 2.0, 5000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _mean_pow3(mid) > P else (lo, mid)
    return ('pow3', 0.5 * (lo + hi))


def _hollow(t, P):
    kind, c = _hollow_coeffs(float(P))
    if kind == 'pow':
        return 1.0 - t**c
    if kind == 'mix':
        return c * (1.0 - t**3)**2 + (1.0 - c) * (1.0 - t**3)
    return (1.0 - t**3)**c


def _phi(t, e):
    """Normalised convex ramp ((t^4 + e^4)^(1/4) - e) / ((1 + e^4)^(1/4) - e): phi'(0) = phi''(0) = 0."""
    return ((t**4 + e**4)**0.25 - e) / ((1.0 + e**4)**0.25 - e)


@lru_cache(maxsize=1)
def _convex_table():
    """Table ln(e) -> mean of 1 - phi(t, e), monotonically increasing from 0.5 (e -> 0) to 0.8 (e -> inf)."""
    le = np.linspace(math.log(1e-4), math.log(1e3), 600)
    t = np.linspace(0.0, 1.0, 4001)
    mean = np.array([1.0 - np.trapezoid(_phi(t, math.exp(x)), t) for x in le])
    return le, mean


def _convex(t, P):
    """Convex branch with mean P (P >= 0.8: 1 - t^n as in the S-shaped family)."""
    if P >= 0.8:
        return 1.0 - t**(P / (1.0 - P))
    le, mean = _convex_table()
    e = math.exp(np.interp(max(P, CONVEX_MIN_P + 1e-4), mean, le))
    return 1.0 - _phi(t, e)


def _branch(t, P, lam):
    """
    SAC branch with mean P: (1 - lam_eff) * S-shaped + lam_eff * convex. Below
    CONVEX_MIN_P + CONVEX_RAMP the convex weight is reduced continuously down to 0 at
    CONVEX_MIN_P (a convex branch with zero slope at the midship section cannot have
    mean < 0.5; near 0.5 it has a narrow shoulder at the end of the parallel middle body).
    """
    t = np.clip(t, 0.0, 1.0)
    lam_eff = lam * _smootherstep((P - CONVEX_MIN_P) / CONVEX_RAMP)
    out = _hollow(t, P)
    if lam_eff > 0.0:
        out = (1.0 - lam_eff) * out + lam_eff * _convex(t, P)
    return out


def bow_lambda(p):
    """
    Weight of the convex family in the forward branch from bow_convexity: the upper half of the
    range (bow_convexity >= 0.5) is fully convex (straight entrance, no inflection); below 0.5 the
    branch blends smoothly (C2) towards the hollow family, which is reached at bow_convexity = 0.
    Any weight below 1 leaves an inflection, so the hollow, small-angle bow is an option of the
    lower half of the range instead of the rule.
    """
    return float(_smootherstep(min(p['bow_convexity'] / 0.5, 1.0)))


def _sac_eval(xi, xa, xf, a_T, F_aft, cp_fwd, lam_fwd):
    """a(xi): convex run branch (up to the transom, a -> a_T), parallel middle body [xa, xf], forebody branch."""
    aft = a_T + (1.0 - a_T) * _branch((xa - xi) / xa, F_aft, 1.0)
    fwd = _branch((xi - xf) / (1.0 - xf), cp_fwd, lam_fwd)
    return np.where(xi < xa, aft, np.where(xi > xf, fwd, 1.0))


def _sac_lcb(cp, pa, xa, xf, a_T, lam_fwd, xi=np.linspace(0.0, 1.0, 2001)):
    pf = (cp - (xf - xa) - xa * pa) / (1.0 - xf)
    a = _sac_eval(xi, xa, xf, a_T, (pa - a_T) / (1.0 - a_T), pf, lam_fwd)
    return np.trapezoid(a * xi, xi) / np.trapezoid(a, xi) - 0.5


def _pa_of(cp, pf, xm, l):
    """Run-branch prismatic giving cp with forebody prismatic pf and midship section at xm."""
    xa, xf = xm - 0.5 * l, xm + 0.5 * l
    return (cp - l - (1.0 - xf) * pf) / xa


def _lcb_curve(cp, pf, l, a_T, lam_fwd):
    """LCB as a function of x_max (grid over XM_RANGE) for given cp and pf; nan where Cp_aft leaves CP_AFT_RANGE."""
    xs = np.linspace(XM_RANGE[0], XM_RANGE[1], 23)
    pa_lo = max(a_T + 0.02, CP_AFT_RANGE[0])
    out = np.full(xs.size, np.nan)
    for i, xm in enumerate(xs):
        pa = _pa_of(cp, pf, xm, l)
        if pa_lo <= pa <= CP_AFT_RANGE[1]:
            out[i] = _sac_lcb(cp, pa, xm - 0.5 * l, xm + 0.5 * l, a_T, lam_fwd)
    return xs, out


def _reachable(cp, lcb, pf, l, a_T, lam_fwd):
    xs, v = _lcb_curve(cp, pf, l, a_T, lam_fwd)
    ok = np.isfinite(v)
    return ok.any() and v[ok].min() <= lcb <= v[ok].max()


def _pf_interval(cp, lcb, l, a_T, lam_fwd):
    """Interval of Cp_fwd (within CP_FWD_RANGE) for which (cp, lcb) can be reached by varying x_max; None if empty."""
    grid = np.linspace(CP_FWD_RANGE[0], CP_FWD_RANGE[1], 15)
    flags = [_reachable(cp, lcb, pf, l, a_T, lam_fwd) for pf in grid]
    if not any(flags):
        return None
    i0 = flags.index(True)
    i1 = len(flags) - 1 - flags[::-1].index(True)

    def edge(good, bad):
        for _ in range(12):
            mid = 0.5 * (good + bad)
            good, bad = (mid, bad) if _reachable(cp, lcb, mid, l, a_T, lam_fwd) else (good, mid)
        return good

    lo = grid[i0] if i0 == 0 else edge(grid[i0], grid[i0 - 1])
    hi = grid[i1] if i1 == len(grid) - 1 else edge(grid[i1], grid[i1 + 1])
    return lo, hi


def _solve_xm(cp, lcb, pf, l, a_T, lam_fwd):
    """x_max giving lcb (LCB increasing with x_max): bisection within the admissible part of the grid."""
    xs, v = _lcb_curve(cp, pf, l, a_T, lam_fwd)
    ok = np.nonzero(np.isfinite(v))[0]
    lo, hi = xs[ok[0]], xs[ok[-1]]
    for i in ok[:-1]:
        if np.isfinite(v[i + 1]) and v[i] <= lcb <= v[i + 1]:
            lo, hi = xs[i], xs[i + 1]
            break

    def f(xm):
        return _sac_lcb(cp, _pa_of(cp, pf, xm, l), xm - 0.5 * l, xm + 0.5 * l, a_T, lam_fwd) - lcb

    f_lo = f(lo)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        lo, hi, f_lo = (mid, hi, f_mid) if f_lo * f_mid > 0.0 else (lo, mid, f_lo)
    return 0.5 * (lo + hi)


@lru_cache(maxsize=256)
def _sac_layout(cp, lcb, fullness, l_max, a_T, lam_fwd):
    """
    Layout of the sectional area curve for prescribed Cp and LCB, with realistic branch prismatics
    (Cp_fwd in CP_FWD_RANGE, Cp_aft in CP_AFT_RANGE) and midship section within XM_RANGE:
    - Cp_fwd = lower bound + fullness (upper bound - lower bound) of the interval of
      Cp_fwd for which (Cp, LCB) can be reached (bow_fullness: 0 = finest admissible bow);
    - x_max solved so that the LCB is exact (LCB grows monotonically with x_max), Cp_aft from Cp;
    - parallel middle body l_max, shortened (bisection) only if (Cp, LCB) cannot be reached with l_max;
    - if they cannot be reached even without a parallel middle body (only with the implicit bulb
      and an extreme LCB), the admissible point with the closest LCB is taken (cp brought into the
      reachable interval).
    Returns (l, x_max, Cp_aft, Cp_fwd, cp, lcb).
    """
    l = l_max
    r = _pf_interval(cp, lcb, l, a_T, lam_fwd)
    if r is None and l_max > 0.0:
        if _pf_interval(cp, lcb, 0.0, a_T, lam_fwd) is not None:
            lo, hi = 0.0, l_max
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if _pf_interval(cp, lcb, mid, a_T, lam_fwd) is not None else (lo, mid)
            l = lo
            r = _pf_interval(cp, lcb, l, a_T, lam_fwd)
        else:
            l = 0.0
    if r is not None:
        pf = r[0] + fullness * (r[1] - r[0])
        xm = _solve_xm(cp, lcb, pf, l, a_T, lam_fwd)
        return l, xm, _pa_of(cp, pf, xm, l), pf, cp, lcb
    # fallback: closest admissible point (l = 0)
    pa_lo = max(a_T + 0.02, CP_AFT_RANGE[0])
    best = None
    for pf in np.linspace(CP_FWD_RANGE[0], CP_FWD_RANGE[1], 15):
        for xm in np.linspace(XM_RANGE[0], XM_RANGE[1], 23):
            c = min(max(cp, xm * pa_lo + (1.0 - xm) * pf), xm * CP_AFT_RANGE[1] + (1.0 - xm) * pf)
            pa = _pa_of(c, pf, xm, 0.0)
            b = _sac_lcb(c, pa, xm, xm, a_T, lam_fwd)
            d = (c - cp)**2 * 100.0 + (b - lcb)**2
            if best is None or d < best[0]:
                best = (d, xm, pa, pf, c, b)
    return (0.0,) + best[1:]


def _form_core(p, cp, lcb):
    """
    Form quantities of the bare hull for prismatic cp and LCB lcb (see form_data and _sac_layout):
    x_max is solved for, the parallel middle body is at most L_pmb.
    """
    rho_T = p['transom_imm'] * p['transom_beam']
    a_T = rho_T * p['c_transom'] / p['Cm']
    l_pmb, xm, cp_aft, cp_fwd, cp, lcb = _sac_layout(float(cp), float(lcb), float(p['bow_fullness']),
                                                     float(p['L_pmb']), float(a_T), bow_lambda(p))
    # Entrance angle: at the FP b = B_half rho, rho = a Cm / c_bow (a -> 0), hence
    # tan(i_E) = B_half Cm |a'(1)| / (c_bow L). With c_bow in C_BOW_RANGE the reachable angle goes from
    # i_lo (maximum c_bow) to i_hi (minimum c_bow); entrance_angle selects the position within the interval
    # (in degrees) and c_bow is derived from it. With a hollow bow (a'(1) = 0) the angle is zero and c_bow
    # goes from C_BOW_RANGE[1] to C_BOW_RANGE[0] with entrance_angle.
    xf = xm + 0.5 * l_pmb
    h = 1e-5
    da = (_sac_eval(np.array(1.0 - h), xm - 0.5 * l_pmb, xf, a_T, (cp_aft - a_T) / (1.0 - a_T), cp_fwd,
                    bow_lambda(p)) / h).item()
    B_half = dimensions(p)[0]
    k_e = B_half * p['Cm'] * da / p['L']
    i_lo = math.atan(k_e / C_BOW_RANGE[1])
    i_hi = math.atan(k_e / C_BOW_RANGE[0])
    e = p['entrance_angle']
    if k_e > 1e-9:
        i_e = i_lo + e * (i_hi - i_lo)
        c_bow = k_e / math.tan(i_e)
    else:
        i_e = 0.0
        c_bow = C_BOW_RANGE[1] - e * (C_BOW_RANGE[1] - C_BOW_RANGE[0])
    return {
        'a_T': a_T, 'rho_T': rho_T, 'xa': xm - 0.5 * l_pmb, 'xf': xf,
        'x_max': xm, 'L_pmb_eff': l_pmb,
        'c_bow': c_bow, 'i_E': math.degrees(i_e), 'i_E_min': math.degrees(i_lo), 'i_E_max': math.degrees(i_hi),
        'beta_aft': math.log(p['transom_imm']) / math.log(rho_T),
        'Cp_bare': cp, 'LCB_bare': lcb,
        'Cp_aft': cp_aft, 'Cp_fwd': cp_fwd,
        'F_aft': (cp_aft - a_T) / (1.0 - a_T),
    }


_DERIVED_CACHE = {}


def form_data(p):
    """
    Derived form quantities, constructed so as to be always feasible:
    - transom: rho_T = (t_T / T)(b_T / B_half), area a_T = A_T / A_M = rho_T c_T / Cm;
    - run (afterbody): t / T = rho^beta, b / B_half = rho^(1 - beta), beta from beta_aft at the transom,
      see _run_deficit (keel_rise_short), beta_aft = ln(t_T / T) / ln(rho_T);
    - x_max and branch prismatics Cp_aft, Cp_fwd from Cp, LCB and bow_fullness (_sac_layout);
    - implicit bulb (bulb_implicit = w): Cp and LCB of the bare hull are reduced by the volume
      (and moment) added by bulb and stem blend, multiplied by w,
      so that with w = 1 the complete hull has exactly the prescribed Cp and LCB;
    - bulb half-breadth b_bulb calibrated so that the maximum breadth of the section
      at the FP below H_B is B_B = C_BB B_MS; if the hull at the FP is already almost as wide as
      B_B, b_bulb = BULB_MIN_AMP B_B / 2 and the resulting breadth (B_B_target) is larger than B_B.
    The result is cached per parameter set.
    """
    if '_bt_corr' not in p:
        return form_data(dict(p, _bt_corr=_bt_correction(p)))
    key = tuple(float(p[n]) for n in ALL_NAMES) + (float(p['_bt_corr']),)
    if key in _DERIVED_CACHE:
        return _DERIVED_CACHE[key]
    L = p['L']
    B_half, T, _ = dimensions(p)
    A_M = p['Cm'] * 2.0 * B_half * T
    B_B, L_PR, Z_B, H_B = bulb_dimensions(p)
    target = 0.5 * B_B
    w = p['bulb_implicit']
    cp_b, lcb_b = p['Cp'], p['LCB']
    b_bulb = target
    fd = None
    for _ in range(8):
        fd = dict(_form_core(p, cp_b, lcb_b), b_bulb=b_bulb)
        # calibration of the bulb breadth at the FP
        meas = 0.5 * _fp_max_breadth(p, fd)
        y_hull_sq = max(meas**2 - b_bulb**2, 0.0)
        b_new = math.sqrt(max(target**2 - y_hull_sq, (BULB_MIN_AMP * target)**2))
        # volume and moment added with respect to the bare-hull SAC
        cp_new, lcb_new = p['Cp'], p['LCB']
        if w > 0.0:
            vol, mom = _volume_moment(p, dict(fd, b_bulb=b_new))
            d_vol = vol - cp_b * A_M * L
            d_mom = mom - cp_b * lcb_b * A_M * L * L
            cp_new = p['Cp'] - w * d_vol / (A_M * L)
            lcb_new = (p['Cp'] * p['LCB'] - w * d_mom / (A_M * L * L)) / cp_new
        done = (abs(b_new - b_bulb) < 1e-5 * L and abs(cp_new - cp_b) < 1e-6 and abs(lcb_new - lcb_b) < 1e-6)
        b_bulb, cp_b, lcb_b = b_new, cp_new, lcb_new
        if done:
            break
    fd = dict(_form_core(p, cp_b, lcb_b), b_bulb=b_bulb,
              B_B_target=2.0 * math.sqrt(y_hull_sq + b_bulb**2))
    if len(_DERIVED_CACHE) > 64:
        _DERIVED_CACHE.clear()
    _DERIVED_CACHE[key] = fd
    return fd


def sectional_area(p, xi, fd=None):
    """
    Normalised sectional area curve a(xi) = A(x) / A_M, xi = x / L (0 transom, 1 FP),
    of the bare hull only (without bulb or stem blend):
    run branch a = a_T + (1 - a_T) f((xa - xi) / xa) (convex, mean F_aft), parallel
    middle body a = 1 in [xa, xf] (L_pmb), forebody branch a = f((xi - xf) / (1 - xf)) with mean
    Cp_fwd and convexity bow_convexity. The means give exactly Cp_bare and LCB_bare.
    """
    fd = form_data(p) if fd is None else fd
    return _sac_eval(xi, fd['xa'], fd['xf'], fd['a_T'], fd['F_aft'], fd['Cp_fwd'], bow_lambda(p))


# ---------------------------------------------------------------------------
# Section family
# ---------------------------------------------------------------------------

def _section_eta(zeta, k, q):
    """
    Normalised section half-breadth eta = y / b, zeta = (z - z_keel) / t
    (0 keel, 1 waterline): regularised minimum between the bottom eta_f = k zeta (deadrise
    tan(beta) = t / (k b)) and the side eta_s, chosen so that eta(1) = 1:
    eta = ((k zeta)^-q + eta_s^-q)^(-1/q), eta_s = (1 - k^-q)^(-1/q).
    q = bilge exponent (low = soft bilge, high = almost a chine). The
    section coefficient grows from 0.5 (k -> 1, V section) to 1 (k -> inf, flat bottom
    and vertical side). Above the waterline the same formula continues towards eta_s.
    """
    eta_s = (1.0 - k**(-q))**(-1.0 / q)
    with np.errstate(divide='ignore', over='ignore'):
        return np.where(zeta > 0.0, ((k * np.maximum(zeta, 1e-300))**(-q) + eta_s**(-q))**(-1.0 / q), 0.0)


@lru_cache(maxsize=1)
def _section_table():
    """
    Table of ln(k - 1) over SECTION_Q_GRID x c in [0.501, 0.99]: for each q, c is computed on a
    fine grid of s = ln(k - 1) and inverted with a cubic spline (C2), then sampled on a
    regular, not too fine, grid of c, so that the bicubic spline does not reproduce noise.
    """
    from scipy.interpolate import CubicSpline
    s_k = np.linspace(-9.0, 14.0, 500)
    k = 1.0 + np.exp(s_k)
    zeta = np.linspace(0.0, 1.0, 2001)
    c_grid = np.linspace(0.501, 0.99, 250)
    S = np.empty((SECTION_Q_GRID.size, c_grid.size))
    for i, q in enumerate(SECTION_Q_GRID):
        c = np.trapezoid(_section_eta(zeta[None, :], k[:, None], q), zeta, axis=1)
        keep = np.concatenate([[True], np.diff(c) > 1e-12])
        S[i] = CubicSpline(c[keep], s_k[keep])(c_grid)
    return c_grid, S


@lru_cache(maxsize=1)
def _section_spline():
    """Bicubic spline of ln(k - 1) over (q, c): k smooth (C2) in both variables, hence
    no curvature kinks in the waterlines where q or c vary along the hull."""
    from scipy.interpolate import RectBivariateSpline
    c_grid, K = _section_table()
    return RectBivariateSpline(SECTION_Q_GRID, c_grid, K, kx=3, ky=3)


def section_k(c, q):
    """Parameter k of the section family with section coefficient c and exponent q."""
    c_grid, _ = _section_table()
    qq = np.clip(np.asarray(q, dtype=float), SECTION_Q_GRID[0], SECTION_Q_GRID[-1])
    cc = np.clip(np.asarray(c, dtype=float), c_grid[0], c_grid[-1])
    qq, cc = np.broadcast_arrays(qq, cc)
    return 1.0 + np.exp(_section_spline().ev(qq.ravel(), cc.ravel()).reshape(qq.shape))


def load_parameters(filename="hull_params.dat"):
    """
    Reads the shape and dimension parameters from the external ASCII file.
    All names in ALL_NAMES (shape parameters and fixed data) are required; unknown names are an error.
    """
    params = {}
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration file '{filename}' not found!")

    with open(filename, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            if '=' in line:
                val_str, key_str = line.split('=')
                params[key_str.strip()] = float(val_str.strip())

    missing = [n for n in ALL_NAMES if n not in params]
    unknown = [n for n in params if n not in ALL_NAMES]
    if missing or unknown:
        raise ValueError(f"'{filename}': missing parameters {missing}, unknown parameters {unknown}")

    return params


# ---------------------------------------------------------------------------
# C2 blending functions
# All hull transitions go through these functions, so that
# position, tangent and curvature are continuous everywhere (no hidden
# kinks at the threshold points in u and v).
# ---------------------------------------------------------------------------

def _smootherstep(t):
    """Ramp 0 -> 1 with zero first and second derivatives at both ends (C2)."""
    t = np.clip(t, 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t * t)


def _onset(t):
    """Ramp 0 -> 1 with C2 start at t=0 and unit slope at t=1 (replaces the linear ramp)."""
    t = np.clip(t, 0.0, 1.0)
    return t**3 * (3.0 - 2.0 * t)


def _bulb_section(z, H_B, zb_over_hb):
    """
    Normalised bulb shape S(z) in [0, 1], used both for the section half-breadth
    and for the protrusion of the side profile (Kracht 1978, Fig. 2-3):
    S = sqrt(F(zeta)), F = zeta (1 - zeta)^6 / F_max, maximum at zeta = 1/7.
    zeta(z) = z~ / (z~ + k (1 - z~)), z~ = z / H_B, with k such that the maximum falls
    at z = zb_over_hb H_B (type Delta < 0.5, O ~ 0.5, Nabla > 0.5).
    At the keel S ~ sqrt(z) (rounded bulb bottom and foot); at the top S ~ (1 - zeta)^3,
    so the bulb blends into the hull with C2 continuity.
    """
    zm = np.clip(zb_over_hb, 0.05, 0.95)
    k = 6.0 * zm / (1.0 - zm)
    zt = np.clip(z / H_B, 0.0, 1.0)
    zeta = zt / np.maximum(zt + k * (1.0 - zt), 1e-300)
    f_max = (1.0 / 7.0) * (6.0 / 7.0)**6
    return np.sqrt(zeta * (1.0 - zeta)**6 / f_max)


def _smooth_rows(a, sigma):
    """Gaussian filter along axis 0 (edges replicated); preserves monotonicity along axis 1."""
    if sigma <= 0.0:
        return a
    r = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma)**2)
    k /= k.sum()
    padded = np.concatenate([np.repeat(a[:1], r, axis=0), a, np.repeat(a[-1:], r, axis=0)])
    out = np.zeros_like(a)
    for n, w in enumerate(k):
        out += w * padded[n:n + a.shape[0]]
    return out


def _node_cdf(curves, turn_weight, max_ratio, sigma, combine=False):
    """
    Node distribution function along axis 0 for a bundle of polylines (n, m, 3).
    Density = arc length + turn_weight * turning angle (nodes cluster where
    the lines bend: stem, bilge, spray rails), filtered and capped at max_ratio times
    the mean, so that no cell is smaller than about 1/max_ratio of the mean cell.
    combine=True uses a single (mean) distribution for all lines.
    """
    d = np.diff(curves, axis=0)
    seg = np.linalg.norm(d, axis=-1)
    tang = d / np.maximum(seg, 1e-15)[..., None]
    ang = np.arccos(np.clip(np.sum(tang[1:] * tang[:-1], axis=-1), -1.0, 1.0))
    zero = np.zeros((1,) + ang.shape[1:])
    ang_seg = 0.5 * (np.concatenate([zero, ang]) + np.concatenate([ang, zero]))
    dens = (seg / np.maximum(seg.sum(axis=0), 1e-15)
            + turn_weight * ang_seg / np.maximum(ang_seg.sum(axis=0), 1e-15))
    if combine:
        dens = dens.mean(axis=1, keepdims=True)
    dens = _smooth_rows(dens, sigma)
    dens = np.minimum(dens, max_ratio * dens.mean(axis=0))
    cdf = np.concatenate([np.zeros((1,) + dens.shape[1:]), np.cumsum(dens, axis=0)])
    return cdf / cdf[-1]


def bulb_dimensions(p):
    """
    Bulb dimensions [m] from Kracht's linear parameters (T_FP = H_dwl, horizontal keel
    at z = 0 forward of the keel rise; B_MS = 2 B_half):
    B_B = C_BB B_MS, L_PR = C_LPR L, Z_B = C_ZB T_FP (height of the bulb nose; with
    bulb_rise = 0 also of the maximum breadth), H_B = Z_B / zm (total height of the
    bulb section), with zm = Z_B / H_B ranging from zm_lo (bulb_type = 0, Delta type) to
    zm_hi (bulb_type = 1, Nabla type) and zm_lo >= Z_B / (BULB_TOP_MAX T): the top of the
    bulb always stays below the waterline.
    """
    B_half, H_dwl, _ = dimensions(p)
    Z_B = p['C_ZB'] * H_dwl
    zm_lo = max(BULB_ZM_RANGE[0], Z_B / (BULB_TOP_MAX * H_dwl))
    zm_hi = max(BULB_ZM_RANGE[1], zm_lo)
    zm = zm_lo + p['bulb_type'] * (zm_hi - zm_lo)
    return p['C_BB'] * 2.0 * B_half, p['C_LPR'] * p['L'], Z_B, Z_B / zm


def _run_deficit(R, R0, beta, lam):
    """
    Run (afterbody): with rho = a Cm / c = (b / B_half)(t / T) and R = -ln(rho) (from R0 at the transom
    to 0 at the parallel middle body), the share D = -ln(t / T) of the deficit goes to the immersion and R - D
    to the breadth: t = T e^-D, b = B_half rho e^D. D is defined as a function of R,
    D(R) = int_0^R sigma(r) dr with 0 <= sigma <= 1, hence along the hull
    D' = sigma R' with R' <= 0: immersion and breadth both grow monotonically
    from the transom to the parallel middle body (waterlines without waists), b <= B_half, t <= T.
    sigma = (1 - lam) beta + lam s((r - r_s) / w + 1/2): lam = 0 (keel_rise_short) gives D = beta R
    (keel rising over the whole run), lam = 1 a smooth step at r_s =
    (1 - beta) R0: constant breadth (= transom) aft, horizontal keel forward.
    In both cases D(R0) = beta R0, i.e. the transom immersion is exact.
    """
    r_s = (1.0 - beta) * R0
    w = max(min(0.5 * R0, 2.0 * r_s, 2.0 * (R0 - r_s)), 1e-9)
    x = (R - r_s) / w + 0.5
    xc = np.clip(x, 0.0, 1.0)
    G = np.where(x >= 1.0, x - 0.5, xc**4 * (2.5 - 3.0 * xc + xc * xc))   # int smootherstep
    return (1.0 - lam) * beta * R + lam * w * G


def _deck_z(p, u):
    """Deck line: freeboard F_over_L L at the transom, F_bow_over_L L at the bow (regularised u^1.6)."""
    L = p['L']
    T = dimensions(p)[1]
    e = 0.05
    g = ((u * u + e * e)**0.8 - e**1.6) / ((1.0 + e * e)**0.8 - e**1.6)
    return T + L * (p['F_over_L'] + (p['F_bow_over_L'] - p['F_over_L']) * g)


def _hull_surface(p, u_orig, v, fd=None):
    """
    Analytic surface of the half-hull (starboard side, y >= 0).
    u_orig: 0 = Transom, 1 = Bow extremity (bulb nose at the bulb heights); v: 0 = Keel
    1 = Deck.
    Vertical reference: z = 0 horizontal keel, design waterline at z = H_dwl.
    Longitudinal reference: x = 0 transom at the waterline, x = L forward perpendicular
    (intersection of the stem with the design waterline).
    Accepts numpy arrays of any shape (broadcast) and returns x, y, z.

    The bare hull is built from the sectional area curve (sectional_area): at each station
    A = 2 c b t, with
    - section coefficient c linear in a: c = c_bow + (Cm - c_bow) a forward of the
      parallel middle body (c_bow derived from the entrance angle, see _form_core),
      c = c_T + (Cm - c_T)(a - a_T) / (1 - a_T) aft of it; rho = a Cm / c <= 1;
    - waterline half-breadth b and immersion t from rho = (b / B_half)(t / T): t = T forward
      of the run (horizontal keel), split between draft and breadth in the run (_run_deficit),
      so that both grow monotonically to the parallel body; the transom is immersed by t_T;
      b <= B_half and c in (0.5, 1) by construction;
    - section shape (_section_eta) with the parameter k that gives exactly c, bilge
      exponent from bilge_softness (midship section) to bilge_softness_aft (transom).
    fd: derived quantities (default form_data(p)); passed explicitly only by form_data.
    """
    fd = form_data(p) if fd is None else fd
    L = p['L']
    B_half, H_dwl, _ = dimensions(p)
    B_B, L_PR, Z_B, H_B = bulb_dimensions(p)
    Cm, xa, xf = p['Cm'], fd['xa'], fd['xf']
    q_mid, q_aft = p['bilge_softness'], p['bilge_softness_aft']
    R0 = -math.log(fd['rho_T'])

    def _base(u):
        """Bare hull with flare, before stem and bulb: (x, z, y)."""
        x = u * L
        a = sectional_area(p, u, fd)
        c = np.where(u >= xa, fd['c_bow'] + (Cm - fd['c_bow']) * a,
                     p['c_transom'] + (Cm - p['c_transom']) * (a - fd['a_T']) / (1.0 - fd['a_T']))
        rho = np.clip(a * Cm / c, 1e-300, 1.0)
        # split of the area deficit between immersion and breadth (see _run_deficit)
        D = np.where(u < xa, _run_deficit(-np.log(rho), R0, fd['beta_aft'], p['keel_rise_short']), 0.0)
        t = H_dwl * np.exp(-D)
        b = B_half * rho * np.exp(D)
        keel_z = H_dwl - t
        # bilge exponent: bilge_softness_aft at the transom, bilge_softness on the parallel body,
        # softening to Q_BOW at the stem (no knuckle line on the V-shaped forward sections)
        q = np.where(u < xa, q_mid + (q_aft - q_mid) * (1.0 - _smootherstep(u / xa)),
                     q_mid + (Q_BOW - q_mid) * _smootherstep((u - xf) / (1.0 - xf)))

        deck_z = _deck_z(p, u)
        z = keel_z + (deck_z - keel_z) * v
        # above the waterline the section blends (C2) into the vertical side y = b over the first
        # TOPSIDE_BLEND of the freeboard: the topside shape depends only on the flare and
        # follows the waterline (monotonic aft), not on the continuation of the section family
        # In the run the blend height is limited to 0.086 t / eta'(1), eta'(1) = k^-q
        # (section slope at the waterline): the outward overhang stays below ~3% of b
        # even for low, flat sections (transom), where the blend becomes a rounded edge;
        # towards the parallel middle body the limit fades out (regularised minimum, C2 blend in u).
        k_s = section_k(c, q)
        eta = _section_eta((z - keel_z) / np.maximum(t, 1e-12), k_s, q)
        h_full = TOPSIDE_BLEND * np.maximum(deck_z - H_dwl, 1e-12)
        h_cap = 0.086 * t / np.maximum(k_s**(-q), 1e-12)
        h_aft = (h_full**-6 + h_cap**-6)**(-1.0 / 6.0)
        w_run = _smootherstep((u - 0.7 * xa) / (0.3 * xa))
        h_b = h_aft + (h_full - h_aft) * w_run
        w_top = _smootherstep((z - H_dwl) / h_b)
        y = b * ((1.0 - w_top) * eta + w_top)

        # Topside flare (zero below the waterline): forward flare, zero at the stem
        # (sqrt(b)) and aft of the parallel middle body; aft/midship flare (negative = tumblehome),
        # fading out towards the bow
        above = _onset((z - H_dwl) / np.maximum(deck_z - H_dwl, 1e-12))
        fwd_w = _onset((u - xf) / (1.0 - xf))
        y = y + p['flare_over_L'] * L * np.sqrt(b / B_half) * fwd_w * above
        y = y + p['flare_aft_over_L'] * L * (1.0 - _smootherstep((u - xf) / (1.0 - xf))) * above
        return x, z, np.maximum(y, 0.0)

    x, z, y = _base(u_orig)

    # Rounded stem of radius r = stem_radius_over_L L:
    # y^2 = y_hull^2 + 2 r l (1 - e^(-d/l)) w(d), d distance from the stem, l = STEM_ROUND_LEN L.
    # Near the stem y^2 ~ y_hull^2 + 2 r d: with a straight entrance y_hull = d tan(i_E) this is a
    # hyperbola, i.e. a circle of radius r at the stem that converges to the entrance line without
    # crossing it, so the waterline stays straight with a rounded nose (no swelling of the nose,
    # no S shape). w(d) fades the term out between 2 l and 5 l, where the hull is already much
    # wider than the rounding, so the rest of the hull is unchanged.
    r_stem = p['stem_radius_over_L'] * L
    l_round = STEM_ROUND_LEN * L
    d_bow = (1.0 - u_orig) * L
    w_round = 1.0 - _smootherstep((d_bow - 2.0 * l_round) / (3.0 * l_round))
    y = np.sqrt(y * y + 2.0 * r_stem * l_round * (1.0 - np.exp(-d_bow / l_round)) * w_round)

    # Bulbous bow (Kracht 1978) and stem. Side profile: the bow at height z is
    # x_bow(z) = L + (1 - S) R(z) + L_PR S, with S the normalised bulb shape (nose at
    # height Z_B, horizontal keel) and R(z) the stem, which passes through the FP at the waterline:
    # R = tan(stem_rake) h_f (s + stem_curvature B(s)), s = (z - H_dwl) / h_f, h_f = freeboard
    # at the bow, B(s) = s+^3 (1 - s)^2 / 0.03456 (maximum 1 at s = 0.6; zero value and slope at
    # s = 1, C2 at s = 0): the curvature is a fraction of the rake, hence with |stem_curvature|
    # <= 0.25 the stem never turns back (|B'| <= 3.92). stem_curvature > 0 convex stem
    # (bulging forward), < 0 concave (clipper-like); with stem_rake = 0 the stem stays vertical.
    # The waterlines are stretched up to x_bow with a C2 blend over the last
    # BULB_STRETCH_FRAC L.
    # Bulb bent nose-up (bulb_rise): the height of the maximum breadth is Z_B
    # at the nose and drops to Z_B (1 - bulb_rise) further aft (d_u = (1 - u) L > L_PR + l_fair),
    # always within the same height H_B (bulb bottom at the keel and bulb top unchanged).
    # Section: bulb half-breadth y_b = b_bulb S Lambda(d) (b_bulb calibrated in form_data
    # so that the breadth at the FP is B_B), merged with the hull as y = sqrt(y_hull^2 + y_b^2).
    # Lambda(d), with d the distance from the nose along the waterline: rounded nose
    # (Lambda^2 = 1 - (1 - d / L_PR)^3, y^2 ~ d at the nose), blend over bulb_fair_over_L L
    l_fair = p['bulb_fair_over_L'] * L
    bend = _smootherstep((1.0 - u_orig) * L / (L_PR + l_fair))
    # neck fill: aft of the FP, over the fairing length, the top of the bulb section rises from
    # H_B towards BULB_FILL_TOP T while the bulb amplitude fades, so the bulb merges into the hull
    # as a broad, low bulge instead of a lobe with a groove above it (unchanged at the FP)
    # (only the breadth uses the filled shape S_b; the forward stretch of the waterlines uses the
    # original shape S_x, so the surface cannot fold where the fill changes quickly along the hull)
    fill = _smootherstep(((1.0 - u_orig) * L - L_PR) / l_fair)
    H_top = H_B + (max(BULB_FILL_TOP * H_dwl, H_B) - H_B) * fill
    z_m = Z_B * (1.0 - p['bulb_rise'] * bend)
    S_b = _bulb_section(z, H_top, z_m / H_top)
    S_x = _bulb_section(z, H_B, z_m / H_B)
    h_f = p['F_bow_over_L'] * L
    s_st = (z - H_dwl) / h_f
    s_p = np.clip(s_st, 0.0, 1.0)
    # below the waterline the rake fades out (C2 at the waterline) over STEM_FADE_DEPTH of the
    # draft: the stem is vertical where it meets the bulb, without a notch above it
    s_dn = STEM_FADE_DEPTH * H_dwl / h_f
    s_lin = np.where(s_st < 0.0, s_st * (1.0 - _smootherstep(-s_st / s_dn)), s_st)
    stem = np.tan(np.radians(p['stem_rake'])) * h_f * (s_lin + p['stem_curvature'] * s_p**3 * (1.0 - s_p)**2 / 0.03456)
    x_bow = L + (1.0 - S_x) * stem + L_PR * S_x
    g_b = _smootherstep((u_orig - (1.0 - BULB_STRETCH_FRAC)) / BULB_STRETCH_FRAC)
    x = x + g_b * (x_bow - L)
    d_nose = np.maximum(x_bow - x, 0.0)
    lam_sq = ((1.0 - (1.0 - np.clip(d_nose / L_PR, 0.0, 1.0))**3)
              * (1.0 - _smootherstep((d_nose - L_PR) / l_fair)))
    y = np.sqrt(y * y + (fd['b_bulb'] * S_b)**2 * lam_sq)

    # Transom raked by transom_rake (positive = top part towards aft), rotated
    # about its trace on the waterline (x = 0), blended over TRANSOM_BLEND_FRAC L
    x = x - (np.tan(np.radians(p['transom_rake'])) * (z - H_dwl)
             * (1.0 - _smootherstep(u_orig / TRANSOM_BLEND_FRAC)))

    # Closure on the centreline and at the stem
    y = np.where((v <= 0.0) | (u_orig >= 1.0), 0.0, y)

    return np.broadcast_arrays(x, y, z)


def _fp_max_breadth(p, fd, n_u=500, n_v=240):
    """Maximum breadth (both sides) of the section x = L below the bulb top H_B."""
    L = p['L']
    H_B = bulb_dimensions(p)[3]
    u = np.linspace(1.0 - BULB_STRETCH_FRAC - 0.05, 1.0, n_u)
    v = np.linspace(0.0, 1.0, n_v)
    x, y, z = _hull_surface(p, u[:, None], v[None, :], fd)
    y_fp = np.array([np.interp(L, x[:, j], y[:, j]) for j in range(n_v)])
    z_fp = np.array([np.interp(L, x[:, j], z[:, j]) for j in range(n_v)])
    m = z_fp <= H_B
    return 2.0 * y_fp[m].max() if m.any() else 0.0


def _wetted(p, wl, fd, n_u, n_v):
    """
    Wetted hull up to the waterline z = wl, sampled exactly (for each u, z is
    linear in v, hence v in [0, v_wl(u)]). Returns the x, y, z grids, the outward
    vector area of the starboard-side cells (A) with their centroids, and the triangles of the
    closing face of the immersed transom (both sides, normal pointing aft).
    """
    u = np.linspace(0.0, 1.0, n_u)
    s = np.linspace(0.0, 1.0, n_v)
    z_kd = _hull_surface(p, u[:, None], np.array([[0.0, 1.0]]), fd)[2]
    zk, zd = z_kd[:, 0], z_kd[:, 1]
    v_wl = np.clip((wl - zk) / np.maximum(zd - zk, 1e-15), 0.0, 1.0)
    x, y, z = (a.copy() for a in _hull_surface(p, u[:, None], v_wl[:, None] * s[None, :], fd))
    P = np.stack([x, y, z], axis=-1)
    A = 0.5 * np.cross(P[1:, 1:] - P[:-1, :-1], P[1:, :-1] - P[:-1, 1:])
    C = 0.25 * (P[1:, 1:] + P[:-1, :-1] + P[1:, :-1] + P[:-1, 1:])
    # outward orientation: on the starboard side the normal at mid-hull has a positive y component
    mid = (n_u // 2, n_v // 2)
    if A[mid][1] < 0.0:
        A = -A
    # immersed transom face (planar): fan from the polygon centroid
    tri_A, tri_C = np.zeros((0, 3)), np.zeros((0, 3))
    if v_wl[0] > 0.0:
        st = P[0]
        poly = np.concatenate([st, (st * np.array([1.0, -1.0, 1.0]))[::-1]])
        o = poly.mean(axis=0)
        a0, a1 = poly - o, np.roll(poly, -1, axis=0) - o
        tri_A = 0.5 * np.cross(a0, a1)
        tri_C = (poly + np.roll(poly, -1, axis=0) + o) / 3.0
        if tri_A[:, 0].sum() > 0.0:
            tri_A = -tri_A
    return {'u': u, 's': s, 'x': x, 'y': y, 'z': z, 'zk': zk, 'zd': zd, 'v_wl': v_wl,
            'A': A, 'C': C, 'tri_A': tri_A, 'tri_C': tri_C}


def _volume_moment(p, fd, n_u=500, n_v=160):
    """Displaced volume at the design waterline and moment about x = L/2 (divergence theorem)."""
    L = p['L']
    w = _wetted(p, dimensions(p)[1], fd, n_u, n_v)
    xc = np.concatenate([w['C'][..., 0].ravel(), w['tri_C'][:, 0]])
    ax = np.concatenate([2.0 * w['A'][..., 0].ravel(), w['tri_A'][:, 0]])
    vol = np.sum(xc * ax)
    return vol, np.sum(0.5 * xc * xc * ax) - 0.5 * L * vol


def generate_hybrid_hull_control_net(num_u=32, num_v=15, params_file="hull_params.dat",
                                     v_map_smoothing=0.04, params=None):
    """
    Generates the NURBS control net of the hull with bulbous bow.

    The analytic surface is C2 everywhere; the nodes are distributed according to
    arc length + turning angle (with a minimum cell size) and then
    evaluated exactly on the surface:
    - in u the same distribution for all lines (planar sections, x = const.);
    - in v the distribution of each section is filtered along the hull
      (v_map_smoothing = sigma as a fraction of the length), so that the lines
      j = const. do not oscillate from one section to the next.
    """
    p = load_parameters(params_file) if params is None else params

    # 1. Temporary high-resolution grid
    # (in v refined logarithmically towards the keel, where full sections
    # have an almost flat bottom)
    n_dense_u = max(2000, num_u * 16)
    n_dense_v = max(200, num_v * 8)
    u_dense = np.linspace(0.0, 1.0, n_dense_u)
    v_dense = np.unique(np.concatenate([[0.0], np.linspace(0.0, 1.0, n_dense_v),
                                        np.logspace(-40.0, -1.0, n_dense_v)]))

    dense_net = np.stack(_hull_surface(p, u_dense[:, None], v_dense[None, :]), axis=-1)
    h_min = np.min(dense_net[:, -1, 2] - dense_net[:, 0, 2])
    if h_min < 0.1:
        raise ValueError(f"Invalid geometry: minimum section height {h_min:.3f} m "
                         "(check volume, beam_rel, F_over_L)")

    # 2. Common longitudinal distribution
    s_u = _node_cdf(dense_net, turn_weight=0.5, max_ratio=3.0,
                    sigma=0.004 * n_dense_u, combine=True)[:, 0]
    # a share BULB_NODE_FRAC of the nodes is spread evenly over the bulb and its fairing
    # (last L_PR + bulb_fair_over_L L of the hull, measured on the keel line), so the bulb is
    # not faceted on coarse grids; the blend of the two distributions keeps it smooth
    _, L_PR, _, _ = bulb_dimensions(p)
    u_b0 = max(0.5, 1.0 - (L_PR + p['bulb_fair_over_L'] * p['L']) / p['L'])
    s_b = np.clip((u_dense - u_b0) / (1.0 - u_b0), 0.0, 1.0)
    s_u = (1.0 - BULB_NODE_FRAC) * s_u + BULB_NODE_FRAC * s_b
    u_nodes = np.interp(np.linspace(0.0, 1.0, num_u), s_u, u_dense)

    # 3. Vertical distribution per section, filtered along the hull
    # (dense points at negligible distance are discarded: the turning angle
    # on zero-length segments is undefined)
    stations = np.stack(_hull_surface(p, u_nodes[None, :], v_dense[:, None]), axis=-1)
    target_v = np.linspace(0.0, 1.0, num_v)
    v_nodes = np.zeros((num_u, num_v))
    for i in range(num_u):
        seg = np.linalg.norm(np.diff(stations[:, i], axis=0), axis=-1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        keep = np.concatenate([[True], np.diff(np.floor(arc / (1e-4 * arc[-1]))) > 0])
        keep[-1] = True
        s_v = _node_cdf(stations[keep, i][:, None], turn_weight=0.3, max_ratio=3.0,
                        sigma=0.004 * keep.sum())[:, 0]
        v_nodes[i] = np.interp(target_v, s_v, v_dense[keep])
    # filter applied to v**0.1, a variable in which the map is smooth even near the keel
    v_nodes = _smooth_rows(v_nodes ** 0.1, v_map_smoothing * (num_u - 1)) ** 10.0
    v_nodes[:, 0], v_nodes[:, -1] = 0.0, 1.0

    # Minimum cell: the node positions (curvilinear abscissa of the actual section) are
    # blended with the uniform distribution, with a single factor for the whole hull,
    # until no cell is smaller than MIN_CELL_FRAC times the mean cell
    real = np.stack(_hull_surface(p, u_nodes[None, :], v_dense[:, None]), axis=-1)
    arc = np.concatenate([np.zeros((1, num_u)),
                          np.cumsum(np.linalg.norm(np.diff(real, axis=0), axis=-1), axis=0)])
    arc /= arc[-1]
    s_nodes = np.array([np.interp(v_nodes[i], v_dense, arc[:, i]) for i in range(num_u)])
    ds_min = np.diff(s_nodes, axis=1).min()
    ds_mean = 1.0 / (num_v - 1)
    lam = np.clip((MIN_CELL_FRAC * ds_mean - ds_min) / (ds_mean - ds_min), 0.0, 1.0)
    if lam > 0.0:
        s_nodes = (1.0 - lam) * s_nodes + lam * target_v[None, :]
        v_nodes = np.array([np.interp(s_nodes[i], arc[:, i], v_dense) for i in range(num_u)])

    # 4. Exact evaluation of the surface at the nodes
    control_net = np.stack(_hull_surface(p, u_nodes[:, None], v_nodes), axis=-1)
    control_net[-1, :, 1] = 0.0

    return control_net


def compute_hydrostatics(p, n_u=1200, n_v=600, wl=None):
    """
    Upright hydrostatics at the waterline z = wl (default: design waterline, z = H_dwl),
    complete hull (both sides), x measured from the transom towards the bow.

    The sections at bow and stern are not planar, so the volume integrals are evaluated as
    surface integrals (divergence theorem) over the wetted hull closed by the waterplane and the
    immersed transom face: int f'(x) dV = int f(x) n_x dS (volume, LCB, sectional area
    curve, V_PR) and int z dV = int z^2/2 n_z dS (KB).
    Also returns the form coefficients, Kracht's non-linear bulb parameters
    (C_ABT, C_ABL, C_VPR), the stability quantities and the design values.
    """
    L = p['L']
    B_half, T, _ = dimensions(p)
    wl = T if wl is None else wl
    B_B, L_PR, Z_B, H_B = bulb_dimensions(p)
    fd = form_data(p)
    w = _wetted(p, wl, fd, n_u, n_v)
    x, y, zk, zd, s = w['x'], w['y'], w['zk'], w['zd'], w['s']

    # cells of both sides + transom face
    xc = np.concatenate([w['C'][..., 0].ravel(), w['tri_C'][:, 0]])
    zc = np.concatenate([w['C'][..., 2].ravel(), w['tri_C'][:, 2]])
    ax = np.concatenate([2.0 * w['A'][..., 0].ravel(), w['tri_A'][:, 0]])
    az = np.concatenate([2.0 * w['A'][..., 2].ravel(), w['tri_A'][:, 2]])
    volume = np.sum(xc * ax)
    lcb = np.sum(0.5 * xc * xc * ax) / max(volume, 1e-15)
    s_wet = 2.0 * np.sum(np.linalg.norm(w['A'], axis=-1))

    # Sectional area curve: A(X) = sum_{x > X} n_x dS
    order = np.argsort(xc)
    x_sorted = xc[order]
    a_curve = np.cumsum(ax[order][::-1])[::-1]
    a_ms = a_curve.max()
    a_bt = a_curve[np.searchsorted(x_sorted, L)] if x_sorted[-1] > L else 0.0
    v_pr = np.sum(np.maximum(xc - L, 0.0) * ax)

    # Waterplane: waterline point for each u (x monotonic in u)
    x_wl, b_wl = x[:, -1], y[:, -1]
    wet = (zk < wl) & (zd > wl)
    b_wl = np.where(wet, b_wl, 0.0)
    awp = 2.0 * np.trapezoid(b_wl, x_wl)
    lcf = np.trapezoid(b_wl * x_wl, x_wl) * 2.0 / max(awp, 1e-15)
    x_wet = x_wl[wet | (zk < wl)]
    lwl = (x_wet.max() - x_wet.min()) if x_wet.size else 0.0
    i_t = (2.0 / 3.0) * np.trapezoid(b_wl**3, x_wl)
    i_l = 2.0 * np.trapezoid(b_wl * (x_wl - lcf)**2, x_wl)

    # Vertical centre of buoyancy (int z dV = int z^2/2 n_z dS, waterplane included)
    kb = (np.sum(0.5 * zc * zc * az) + 0.5 * wl * wl * awp) / max(volume, 1e-15)
    vol_z = np.sum(zc * az) + wl * awp          # check: same volume with f = z

    # Stem profile (u = 1): lateral area of the bulb forward of the FP
    x_st, _, z_st = _hull_surface(p, 1.0, s * w['v_wl'][-1])
    a_bl = np.trapezoid(np.maximum(x_st - L, 0.0), z_st)

    b_b_eff = _fp_max_breadth(p, fd)
    b_ms = 2.0 * B_half
    y_max = _hull_surface(p, np.linspace(0.0, 1.0, 600)[:, None], np.linspace(0.0, 1.0, 200)[None, :])[1].max()
    t_max = wl - zk.min()
    displacement = RHO_SEA * volume

    # Design values from the bare-hull sectional area curve
    xi = np.linspace(0.0, 1.0, 4001)
    a_d = sectional_area(p, xi, fd)
    k_mid = float(section_k(p['Cm'], p['bilge_softness']))
    # design waterline half-breadth in the forebody (horizontal keel: b = B_half rho)
    xi_f = np.linspace(fd['xf'], 1.0, 4001)
    a_f = sectional_area(p, xi_f, fd)
    b_f = B_half * a_f * p['Cm'] / (fd['c_bow'] + (p['Cm'] - fd['c_bow']) * a_f)
    db = np.gradient(b_f, xi_f * L)
    d2 = np.gradient(db, xi_f * L)
    sig = np.sign(np.where(np.abs(d2) > 1e-3 * np.abs(d2).max(), d2, 0.0))
    sig = sig[sig != 0]

    return {
        "volume": volume,
        "displacement": displacement,
        "LCB": lcb,
        "LCB_frac_L": lcb / L,
        "LWL": lwl,
        "BWL": 2.0 * b_wl.max(),
        "Awp": awp,
        "LCF": lcf,
        "S_wet": s_wet,
        "A_transom": -np.sum(w['tri_A'][:, 0]),
        "T_transom": max(wl - zk[0], 0.0),
        "T_max": t_max,
        "Cb": volume / max(lwl * 2.0 * b_wl.max() * t_max, 1e-15),
        "H_dwl": wl,
        "B_max": 2.0 * y_max,
        "L_over_B_max": L / (2.0 * y_max),
        "H_transom": zd[0] - zk[0],
        # Main dimensions derived from the volume (B maximum beam at the design waterline)
        "volume_target": p['volume'],
        "volume_error": volume / p['volume'] - 1.0,
        "B": 2.0 * B_half,
        "L_over_B": L / (2.0 * B_half),
        "B_over_T": 2.0 * B_half / T,
        # Form coefficients of the generated hull (bulb and stem included)
        "Cp": volume / max(a_ms * L, 1e-15),
        "Cm": a_ms / max(2.0 * b_wl.max() * t_max, 1e-15),
        "Cwp": awp / max(L * 2.0 * b_wl.max(), 1e-15),
        "LCB_mid_frac": lcb / L - 0.5,
        # implicit bulb: deviation of the complete hull from the prescribed values (expected
        # (1 - bulb_implicit) times the bulb effect; different only if the LCB has been limited)
        "Cp_minus_input": volume / max(a_ms * L, 1e-15) - p['Cp'],
        "LCB_minus_input": lcb / L - 0.5 - p['LCB'],
        "A_T_over_A_M": -np.sum(w['tri_A'][:, 0]) / max(a_ms, 1e-15),
        # Stability and ratios
        "KB": kb,
        "BM_T": i_t / max(volume, 1e-15),
        "BM_L": i_l / max(volume, 1e-15),
        "KM_T": kb + i_t / max(volume, 1e-15),
        "I_T": i_t,
        "I_L": i_l,
        "TPC": RHO_SEA * awp / 100.0,
        "MCT_1cm": displacement * i_l / max(volume, 1e-15) / (100.0 * L),
        "vol_over_01L3": volume / (0.1 * L)**3,
        "L_over_vol13": L / max(volume, 1e-15)**(1.0 / 3.0),
        "volume_check_z": vol_z,
        # Design values: bare hull from the sectional area curve (with the implicit bulb Cp_bare
        # and LCB_bare are reduced so that the complete hull has the prescribed Cp and LCB)
        "Cp_bare": np.trapezoid(a_d, xi),
        "LCB_bare_mid_frac": np.trapezoid(a_d * xi, xi) / np.trapezoid(a_d, xi) - 0.5,
        "Cp_aft_bare": fd['Cp_aft'],
        "L_pmb_eff": fd['L_pmb_eff'],
        "x_max": fd['x_max'],
        "Cp_fwd_bare": fd['Cp_fwd'],
        "deadrise_mid_deg": math.degrees(math.atan(T / (k_mid * B_half))),
        "entrance_half_angle_deg": math.degrees(math.atan(abs(db[-2]))),
        "entrance_angle_deg": fd['i_E'],
        "entrance_angle_min_deg": fd['i_E_min'],
        "entrance_angle_max_deg": fd['i_E_max'],
        "c_bow": fd['c_bow'],
        "wl_inflections_fwd": float(np.sum(sig[1:] != sig[:-1])),
        # Bulb (Kracht 1978): actual dimensions and parameters of the generated hull
        "L_PR": L_PR,
        "Z_B": Z_B,
        "H_B": H_B,
        "b_bulb": fd['b_bulb'],
        "C_BB_target": fd['B_B_target'] / b_ms,
        "B_B": b_b_eff,
        "C_BB_eff": b_b_eff / b_ms,
        "A_MS": a_ms,
        "A_BT": a_bt,
        "A_BL": a_bl,
        "V_PR": v_pr,
        "C_ABT": a_bt / max(a_ms, 1e-15),
        "C_ABL": a_bl / max(a_ms, 1e-15),
        "C_VPR": v_pr / max(volume, 1e-15),
    }


def feasibility_checks(p, hydro=None, n_u=500, n_v=160):
    """
    Validity checks of the variant (numerical values, 1 = passed for the boolean
    checks) and overall indicator 'feasible', written to hull_hydro.dat.
    """
    hydro = compute_hydrostatics(p, 600, 300) if hydro is None else hydro
    B_half, T, _ = dimensions(p)
    H_B = bulb_dimensions(p)[3]
    u = np.linspace(0.0, 1.0, n_u)
    v = np.linspace(0.0, 1.0, n_v)
    X, Y, Z = _hull_surface(p, u[:, None], v[None, :])
    finite = bool(np.all(np.isfinite(X)) and np.all(np.isfinite(Y)) and np.all(np.isfinite(Z)))
    ywl = np.array([np.interp(T, Z[i], Y[i]) for i in range(n_u)])
    xs = X[:, 0]
    chk = {
        "chk_finite": float(finite),
        "chk_x_monotone": float(np.all(np.diff(X, axis=0) > -1e-9)),
        "chk_z_monotone": float(np.all(np.diff(Z, axis=1) > -1e-12)),
        "chk_y_nonneg": float(Y.min() >= 0.0),
        # hull waterline half-breadth (aft of the bulb) <= B_half
        "chk_wl_beam": float(ywl[xs < 0.9 * p['L']].max() <= B_half * 1.001),
        "chk_section_height": float(np.min(Z[:, -1] - Z[:, 0]) >= 0.25 * T),
        "chk_bulb_below_wl": float(H_B <= BULB_TOP_MAX * T + 1e-9),
        "chk_C_BB": float(abs(hydro['B_B'] - form_data(p)['B_B_target']) <= 0.01 * 2.0 * B_half),
        "chk_volume_closure": float(abs(hydro['volume_check_z'] / hydro['volume'] - 1.0) < 2e-3),
        "chk_stern_wl_monotone": float(_stern_monotone(p)),
        "chk_cp_fwd": float(form_data(p)['Cp_fwd'] >= CP_FWD_RANGE[0] - 1e-6),
        # volume of the complete hull equal to the given one; L / B and B / T within bounds
        # (fails only if the volume is incompatible with L, Cp and Cm)
        "chk_volume": float(abs(hydro['volume_error']) < 2e-3),
        "chk_proportions": float(not beam_window(p, 2.0 * B_half * T)[2]),
    }
    chk["feasible"] = float(all(val == 1.0 for val in chk.values()))
    return chk


def _stern_monotone(p, n_u=1500):
    """
    Run waterlines without waists: from the waterline up to 95% of the freeboard
    at the transom (8 heights) the half-breadth does not decrease from the transom to the parallel
    middle body beyond a tolerance of 1e-4 B_half per step and 2% B_half overall (flare and tumblehome included).
    """
    B_half, T, _ = dimensions(p)
    fd = form_data(p)
    u = np.linspace(0.02 * fd['xa'], fd['xa'], n_u)
    zk = _hull_surface(p, u, 0.0)[2]
    zd = _hull_surface(p, u, 1.0)[2]
    for zz in T + np.linspace(0.0, 0.95, 8) * p['F_over_L'] * p['L']:
        m = zk < zz
        if m.sum() < 3:
            continue
        x, y, _ = _hull_surface(p, u[m], (zz - zk[m]) / (zd[m] - zk[m]))
        if np.min(np.diff(y)) < -1e-4 * B_half and np.max(np.maximum.accumulate(y) - y) > 0.02 * B_half:
            return False
    return True


def hydrostatic_curves(p, draft_fracs, keys=("volume", "displacement", "LCB", "KB", "LCF", "Awp", "BM_T",
                                             "BM_L", "KM_T", "TPC", "MCT_1cm", "S_wet", "Cb", "Cwp")):
    """Hydrostatic curves: rows (z_wl, quantities) for waterlines z = fraction * H_dwl."""
    T = dimensions(p)[1]
    rows = []
    for f in draft_fracs:
        h = compute_hydrostatics(p, 600, 300, wl=f * T)
        rows.append([f * T] + [h[k] for k in keys])
    return ["z_wl"] + list(keys), rows


def export_hydro_curves(filename, header, rows):
    with open(filename, 'w') as f:
        f.write("# Hydrostatic curves (upright hull); z_wl from the bottom of the horizontal keel\n")
        f.write("# " + " ".join(f"{h:>14s}" for h in header) + "\n")
        for r in rows:
            f.write("  " + " ".join(f"{val:14.6e}" for val in r) + "\n")
    print(f"Hydrostatic curves saved to: {filename}")


def export_hydrostatics(filename, hydro):
    """Writes the hydrostatics in the 'value = name' format (same as hull_params.dat)."""
    with open(filename, 'w') as f:
        f.write("# --- HYDROSTATICS AT THE DESIGN WATERLINE (z = H_dwl) ---\n")
        f.write(f"# x from the transom; displacement with rho = {RHO_SEA} t/m^3\n\n")
        for k, val in hydro.items():
            f.write(f"{val:<14.6f} = {k}\n")
    print(f"Hydrostatics saved to: {filename}")


def export_plot3d_ascii(filename, control_net):
    """Exports the grid in PLOT3D ASCII format for Fortran/CFD codes."""
    grid_f = file_order(control_net)
    ni, nj, _ = grid_f.shape
    nk = 1
    nb = 1

    with open(filename, 'w') as f:
        f.write(f"{nb}\n")
        f.write(f"{ni} {nj} {nk}\n")
        for coord_idx in range(3):
            for j in range(nj):
                row_vals = [f"{grid_f[i, j, coord_idx]:16.8e}" for i in range(ni)]
                f.write(" ".join(row_vals) + "\n")

    print(f"PLOT3D ASCII file saved successfully: {filename}")


def file_order(control_net):
    """Net in the file index order: i bow -> transom, j deck -> keel."""
    return control_net[::-1, ::-1, :]


def nurbs_parameters(net_f):
    """Chord-length parameters (Piegl-Tiller 9.2.1) averaged over all lines of the net, in u and in v."""
    def par(lines):
        seg = np.linalg.norm(np.diff(lines, axis=0), axis=-1)
        t = np.concatenate([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
        return (t / np.maximum(t[-1], 1e-15)).mean(axis=1)
    return par(net_f), par(net_f.transpose(1, 0, 2))


def nurbs_knots(net_f, degree_u=3, degree_v=3):
    """
    Clamped knot vectors for the net net_f (file order): chord-length parameters
    averaged over all lines of the net, then de Boor moving average (Piegl-Tiller 9.8),
    so that the knots follow the (non-uniform) spacing of the net points.
    """
    def knots(par, p):
        n = len(par)
        if not 1 <= p < n:
            raise ValueError(f"Degree {p} is not valid for {n} control points")
        inner = [par[j:j + p].mean() for j in range(1, n - p)]
        return np.concatenate([np.zeros(p + 1), inner, np.ones(p + 1)])

    par_u, par_v = nurbs_parameters(net_f)
    return knots(par_u, degree_u), knots(par_v, degree_v)


def _basis_matrix(knots, p, t):
    """Matrix of the B-spline basis functions N[k, i] = N_i,p(t_k) (Cox-de Boor, t = 1 included)."""
    n = len(knots) - p - 1
    t = np.atleast_1d(t)
    N = np.zeros((t.size, len(knots) - 1))
    for i in range(len(knots) - 1):
        N[:, i] = (knots[i] <= t) & (t < knots[i + 1])
    last = np.nonzero(knots < knots[-1])[0][-1]         # last non-empty interval: t = 1 included
    N[t >= knots[-1], last] = 1.0
    for d in range(1, p + 1):
        Nn = np.zeros((t.size, len(knots) - 1 - d))
        for i in range(len(knots) - 1 - d):
            den1 = knots[i + d] - knots[i]
            den2 = knots[i + d + 1] - knots[i + 1]
            a = (t - knots[i]) / den1 * N[:, i] if den1 > 0 else 0.0
            b = (knots[i + d + 1] - t) / den2 * N[:, i + 1] if den2 > 0 else 0.0
            Nn[:, i] = a + b
        N = Nn
    return N[:, :n]


def bspline_interpolate(net_f, degree_u, degree_v, knots_u, knots_v):
    """
    Control points of the B-spline that INTERPOLATES the net points at their chord-length
    parameters (Piegl-Tiller 9.2.5, tensor product: one linear system in u, then in v).
    Boundaries (keel, stem, deck, transom) are interpolated exactly.
    """
    par_u, par_v = nurbs_parameters(net_f)
    Nu = _basis_matrix(knots_u, degree_u, par_u)
    Nv = _basis_matrix(knots_v, degree_v, par_v)
    R = np.linalg.solve(Nu, net_f.reshape(net_f.shape[0], -1)).reshape(net_f.shape)
    Q = np.linalg.solve(Nv, R.transpose(1, 0, 2).reshape(net_f.shape[1], -1))
    return Q.reshape(net_f.shape[1], net_f.shape[0], 3).transpose(1, 0, 2)


def bspline_evaluate(ctrl, degree_u, degree_v, knots_u, knots_v, tu, tv):
    """Evaluates the B-spline surface (unit weights) on the tu x tv grid."""
    Nu = _basis_matrix(knots_u, degree_u, tu)
    Nv = _basis_matrix(knots_v, degree_v, tv)
    return np.einsum('ai,ijc,bj->abc', Nu, ctrl, Nv)


def nurbs_deviation(net_f, ctrl, degree_u, degree_v, knots_u, knots_v):
    """
    Deviation of the B-spline (control points ctrl) from the net points net_f, which
    lie on the analytic surface: distance at the chord-length parameters of the points (maximum and mean).
    """
    par_u, par_v = nurbs_parameters(net_f)
    S = bspline_evaluate(ctrl, degree_u, degree_v, knots_u, knots_v, par_u, par_v)
    d = np.linalg.norm(S - net_f, axis=-1)
    return d.max(), d.mean()


def export_nurbs_ascii(filename, net_f, degree_u, degree_v, knots_u, knots_v):
    """
    Exports degree and knot vectors of the B-spline surface (NURBS with unit weights)
    whose control points are those of the PLOT3D file, in the same index order.
    Format (readable list-directed in Fortran):
        line 1: degree_u degree_v        (u = index i, v = index j of the PLOT3D file)
        line 2: n_ctrl_u n_ctrl_v        (= ni nj)
        line 3: n_ctrl_u + degree_u + 1 knots in u
        line 4: n_ctrl_v + degree_v + 1 knots in v
    """
    ni, nj, _ = net_f.shape
    with open(filename, 'w') as f:
        f.write(f"{degree_u} {degree_v}\n")
        f.write(f"{ni} {nj}\n")
        f.write(" ".join(f"{k:.15e}" for k in knots_u) + "\n")
        f.write(" ".join(f"{k:.15e}" for k in knots_v) + "\n")
    print(f"NURBS degree and knots saved to: {filename}")


def export_step(filename, net_f, degree_u, degree_v, knots_u, knots_v, mirror=True, closed=True):
    """
    Exports the hull to STEP AP214 (ISO 10303-21). Indices as in the files: i = 1 bow ... ni
    transom, j = 1 deck ... nj keel; P(i, j) = net_f[i-1, j-1].
    - Hull: one B_SPLINE_SURFACE_WITH_KNOTS face per side, with exact B-spline edges.
    - closed=True adds, for each side, the closing faces:
      deck: ruled surface (degrees p x 1) between the line j = 1 and its projection on the
      centreplane (same x and z, y = 0), from P(1,1) to the point C = (x, 0, z) of P(ni,1)
      (horizontal in each section, follows the sheer; it is a plane only if the deck is flat);
      transom: PLANE bounded by the line i = ni and by the segments P(ni,1)-C and
      P(ni,nj)-C (the transom is planar by construction: x depends only on z).
      With mirror=True the faces of the two sides share the edges on the centreplane and the
      result is a closed solid (MANIFOLD_SOLID_BREP); with mirror=False an open shell.
    Each loop is traversed counter-clockwise as seen from outside; same_sense of each face is
    chosen by comparing the surface normal with the outward normal.
    """
    ents = []

    def add(s):
        ents.append(s)
        return f"#{len(ents)}"

    def real(a):
        return f"{a:.12E}"

    def point(c):
        return add(f"CARTESIAN_POINT('',({real(c[0])},{real(c[1])},{real(c[2])}))")

    def knot_data(k):
        vals, mults = [], []
        for t in k:
            if vals and abs(t - vals[-1]) < 1e-12:
                mults[-1] += 1
            else:
                vals.append(t)
                mults.append(1)
        return "(" + ",".join(map(str, mults)) + ")", "(" + ",".join(real(t) for t in vals) + ")"

    def curve(pts, p, k):
        m, kv = knot_data(k)
        refs = ",".join(point(c) for c in pts)
        return add(f"B_SPLINE_CURVE_WITH_KNOTS('',{p},({refs}),.UNSPECIFIED.,.F.,.F.,{m},{kv},.UNSPECIFIED.)")

    def surface(P):
        mu, ku = knot_data(knots_u)
        mv, kv = knot_data(knots_v)
        rows = ",".join("(" + ",".join(point(c) for c in row) + ")" for row in P)
        return add(f"B_SPLINE_SURFACE_WITH_KNOTS('',{degree_u},{degree_v},({rows}),.UNSPECIFIED.,"
                   f".F.,.F.,.F.,{mu},{mv},{ku},{kv},.UNSPECIFIED.)")

    def vertex(c):
        return add(f"VERTEX_POINT('',{point(c)})")

    def edge(v1, v2, crv):
        return add(f"EDGE_CURVE('',{v1},{v2},{crv},.T.)")

    def line(a, b):
        return curve(np.array([a, b]), 1, np.array([0.0, 0.0, 1.0, 1.0]))

    def oriented(e, sense):
        return add(f"ORIENTED_EDGE('',*,*,{e},{'.T.' if sense else '.F.'})")

    def adv_face(surf, edges, same_sense):
        """edges: [(edge, sense)] counter-clockwise as seen from outside."""
        loop = add(f"EDGE_LOOP('',({','.join(oriented(e, sd) for e, sd in edges)}))")
        bound = add(f"FACE_OUTER_BOUND('',{loop},.T.)")
        return add(f"ADVANCED_FACE('',({bound}),{surf},{'.T.' if same_sense else '.F.'})")

    def direction(d):
        d = np.asarray(d, dtype=float) / np.linalg.norm(d)
        return add(f"DIRECTION('',({real(d[0])},{real(d[1])},{real(d[2])}))")

    ctx_app = add("APPLICATION_CONTEXT('automotive design')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{ctx_app})")
    ctx_prod = add(f"PRODUCT_CONTEXT('',{ctx_app},'mechanical')")
    prod = add(f"PRODUCT('BulbousBowHull','BulbousBowHull','',({ctx_prod}))")
    pdf = add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
    ctx_pd = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{ctx_app},'design')")
    pd = add(f"PRODUCT_DEFINITION('design','',{pdf},{ctx_pd})")
    pds = add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
    u_len = add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))")
    u_ang = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    u_sol = add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),{u_len},'distance_accuracy_value','')")
    ctx_geo = add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc}))"
                  f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({u_len},{u_ang},{u_sol}))REPRESENTATION_CONTEXT('',''))")
    origin = add(f"AXIS2_PLACEMENT_3D('',{point((0.0, 0.0, 0.0))},$,$)")

    # Orientation of the normal S_u x S_v at mid-surface: outward if y > 0 on the starboard side
    P = net_f
    ni, nj = P.shape[0], P.shape[1]
    mi, mj = ni // 2, nj // 2
    n_y = np.cross(P[mi + 1, mj] - P[mi - 1, mj], P[mi, mj + 1] - P[mi, mj - 1])[1]
    sense_stbd = n_y > 0.0
    flip = np.array([1.0, -1.0, 1.0])
    Q = P * flip

    # Vertices: bow deck/keel and transom keel on the centreplane, transom deck per side,
    # C = centreplane point at the height of P(ni,1) (aft end of the deck line
    # on the centreplane, opposite to P(1,1))
    p_c = np.array([P[-1, 0, 0], 0.0, P[-1, 0, 2]])
    v_bd, v_bk, v_tk = vertex(P[0, 0]), vertex(P[0, -1]), vertex(P[-1, -1])
    v_td_s, v_td_p, v_c = vertex(P[-1, 0]), vertex(Q[-1, 0]), vertex(p_c)

    # Edges (definition direction given in the names)
    e_keel = edge(v_bk, v_tk, curve(P[:, -1], degree_u, knots_u))      # bow -> transom
    e_stem = edge(v_bd, v_bk, curve(P[0, :], degree_v, knots_v))       # deck -> keel
    e_deck_s = edge(v_bd, v_td_s, curve(P[:, 0], degree_u, knots_u))   # bow -> transom
    e_tr_s = edge(v_td_s, v_tk, curve(P[-1, :], degree_v, knots_v))    # deck -> keel
    hull_s = surface(P)
    # starboard hull (outward normal +y): P(1,1) -> stem -> keel -> transom -> deck
    faces = [adv_face(hull_s, [(e_stem, True), (e_keel, True), (e_tr_s, False), (e_deck_s, False)],
                      sense_stbd)]

    if closed:
        cl_deck = P[:, 0] * np.array([1.0, 0.0, 1.0])
        e_cl_deck = edge(v_bd, v_c, curve(cl_deck, degree_u, knots_u))   # bow -> transom, y = 0
        e_top_s = edge(v_td_s, v_c, line(P[-1, 0], p_c))
        e_cl_tr = edge(v_tk, v_c, line(P[-1, -1], p_c))
        # deck: ruled surface between the deck line and its projection on the centreplane
        mv, kv = knot_data(np.array([0.0, 0.0, 1.0, 1.0]))
        mu, ku = knot_data(knots_u)

        def deck_surface(edge_pts):
            rows = ",".join("(" + point(a) + "," + point(b) + ")" for a, b in zip(edge_pts, cl_deck))
            return add(f"B_SPLINE_SURFACE_WITH_KNOTS('',{degree_u},1,({rows}),.UNSPECIFIED.,"
                       f".F.,.F.,.F.,{mu},{mv},{ku},{kv},.UNSPECIFIED.)")

        def deck_up(edge_pts):
            # normal S_u x S_v of the ruled surface at mid-length: must point towards +z
            k = len(edge_pts) // 2
            su = edge_pts[k + 1] - edge_pts[k - 1]
            sv = cl_deck[k] - edge_pts[k]
            return np.cross(su, sv)[2] > 0.0

        # transom: plane through the transom points, outward normal pointing aft
        n_tr = np.cross([0.0, 1.0, 0.0], P[-1, -1] - p_c)
        n_tr = -n_tr if n_tr[0] > 0.0 else n_tr
        x_tr = np.cross([0.0, 1.0, 0.0], n_tr)
        plane_tr = add(f"PLANE('',{add(f'AXIS2_PLACEMENT_3D({chr(39)}{chr(39)},{point(p_c)},{direction(n_tr)},{direction(x_tr)})')})")

        # starboard deck (normal +z): C -> P(1,1) along the centreplane, deck, P(ni,1) -> C
        faces.append(adv_face(deck_surface(P[:, 0]), [(e_cl_deck, False), (e_deck_s, True), (e_top_s, True)],
                              deck_up(P[:, 0])))
        # starboard transom (normal -x): P(ni,1) -> P(ni,nj) -> C -> P(ni,1)
        faces.append(adv_face(plane_tr, [(e_tr_s, True), (e_cl_tr, True), (e_top_s, False)], True))

    if mirror:
        # Port side: mirrored in y; keel, stem and centreplane edges are shared
        e_deck_p = edge(v_bd, v_td_p, curve(Q[:, 0], degree_u, knots_u))
        e_tr_p = edge(v_td_p, v_tk, curve(Q[-1, :], degree_v, knots_v))
        # port hull (normal -y): P(1,1) -> deck -> transom -> keel -> stem
        faces.append(adv_face(surface(Q), [(e_deck_p, True), (e_tr_p, True), (e_keel, False), (e_stem, False)],
                              not sense_stbd))
        if closed:
            e_top_p = edge(v_td_p, v_c, line(Q[-1, 0], p_c))
            faces.append(adv_face(deck_surface(Q[:, 0]), [(e_deck_p, False), (e_cl_deck, True), (e_top_p, False)],
                                  deck_up(Q[:, 0])))
            faces.append(adv_face(plane_tr, [(e_cl_tr, False), (e_tr_p, False), (e_top_p, True)], True))

    if closed and mirror:
        shell = add(f"CLOSED_SHELL('',({','.join(faces)}))")
        brep = add(f"MANIFOLD_SOLID_BREP('BulbousBowHull',{shell})")
        rep = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('BulbousBowHull',({origin},{brep}),{ctx_geo})")
    else:
        shell = add(f"OPEN_SHELL('',({','.join(faces)}))")
        model = add(f"SHELL_BASED_SURFACE_MODEL('',({shell}))")
        rep = add(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('BulbousBowHull',({origin},{model}),{ctx_geo})")
    add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{rep})")

    with open(filename, 'w') as f:
        f.write("ISO-10303-21;\nHEADER;\n")
        f.write("FILE_DESCRIPTION(('Parametric hull with bulbous bow (B-spline)'),'2;1');\n")
        f.write(f"FILE_NAME('{os.path.basename(filename)}','',(''),(''),'gen_hull.py','gen_hull.py','');\n")
        f.write("FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n")
        for n, s in enumerate(ents, start=1):
            f.write(f"#{n}={s};\n")
        f.write("ENDSEC;\nEND-ISO-10303-21;\n")
    print(f"STEP file saved to: {filename}")


def plot_hull_3d(control_net, show_control_points=True, mirror_port_side=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = control_net[:, :, 0], control_net[:, :, 1], control_net[:, :, 2]

    ax.plot_surface(X, Y, Z, cmap='ocean', edgecolor='k', linewidth=0.3, alpha=0.92, antialiased=True)
    if mirror_port_side:
        ax.plot_surface(X, -Y, Z, cmap='ocean', edgecolor='k', linewidth=0.3, alpha=0.92, antialiased=True)

    if show_control_points:
        ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), color='red', s=10, label='Control points', depthshade=False)

    max_range = np.array([
        X.max() - X.min(),
        (Y.max() * 2 if mirror_port_side else Y.max()) - (-Y.max() if mirror_port_side else 0),
        Z.max() - Z.min()
    ]).max() / 2.0

    mid_x, mid_y, mid_z = (X.max() + X.min()) * 0.5, 0.0, (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X - Length")
    ax.set_ylabel("Y - Breadth")
    ax.set_zlabel("Z - Height")
    ax.set_title("Motor-yacht hull with bulbous bow")
    ax.view_init(elev=15, azim=-115)
    if show_control_points:
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametric motor-yacht hull generator with bulbous bow for PLOT3D / Fortran")
    parser.add_argument("--no-plot", action="store_true", help="Disable the 3D plot window")
    parser.add_argument("--params", type=str, default="hull_params.dat", help="Path of the parameter file")
    parser.add_argument("--output", type=str, default="hull_grid.p3d", help="Output PLOT3D file name")
    parser.add_argument("--hydro", type=str, default="hull_hydro.dat", help="Output hydrostatics file name")
    parser.add_argument("--hydro-curves", type=str, default=None,
                        help="If given, write the hydrostatic curves to this file")
    parser.add_argument("--drafts", type=str, default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2",
                        help="Waterlines of the hydrostatic curves (fractions of H_dwl, comma-separated)")
    parser.add_argument("--ni", type=int, default=32, help="Number of longitudinal nodes i")
    parser.add_argument("--nj", type=int, default=15, help="Number of transverse nodes j")
    parser.add_argument("--v-smooth", type=float, default=0.04,
                        help="Smoothing of the vertical node distribution along the hull (fraction of L, 0 = off)")
    parser.add_argument("--degree", type=int, default=3, help="Degree of the B-spline surface in u and v")
    parser.add_argument("--nurbs", type=str, default="hull_nurbs.dat", help="Output NURBS degree/knots file name")
    parser.add_argument("--nurbs-fit", choices=("approx", "interp"), default="approx",
                        help="approx: the net points are the control points (the B-spline approximates the "
                             "surface); interp: control points computed so that the B-spline passes through "
                             "the net points (PLOT3D and STEP then contain the control points)")
    parser.add_argument("--step", type=str, default="hull.step", help="Output STEP file name")
    parser.add_argument("--step-half", action="store_true", help="STEP with the starboard side only (default: complete hull)")
    parser.add_argument("--step-open", action="store_true",
                        help="STEP with the hull surface only, without deck and transom closures (default: closed solid)")

    args = parser.parse_args()

    params = load_parameters(args.params)
    hull_net = generate_hybrid_hull_control_net(num_u=args.ni, num_v=args.nj, params=params,
                                                v_map_smoothing=args.v_smooth)
    net_f = file_order(hull_net)
    knots_u, knots_v = nurbs_knots(net_f, args.degree, args.degree)
    ctrl_f = net_f
    if args.nurbs_fit == "interp":
        ctrl_f = bspline_interpolate(net_f, args.degree, args.degree, knots_u, knots_v)
    dev_max, dev_mean = nurbs_deviation(net_f, ctrl_f, args.degree, args.degree, knots_u, knots_v)
    print(f"B-spline ({args.nurbs_fit}): deviation from the surface points max {dev_max:.3e} m, "
          f"mean {dev_mean:.3e} m")

    export_plot3d_ascii(args.output, file_order(ctrl_f))
    export_nurbs_ascii(args.nurbs, ctrl_f, args.degree, args.degree, knots_u, knots_v)
    export_step(args.step, ctrl_f, args.degree, args.degree, knots_u, knots_v, mirror=not args.step_half,
                closed=not args.step_open)

    hydro = compute_hydrostatics(params)
    hydro.update(feasibility_checks(params, hydro))
    hydro["nurbs_dev_max"], hydro["nurbs_dev_mean"] = dev_max, dev_mean
    export_hydrostatics(args.hydro, hydro)
    print(f"Volume {hydro['volume']:.3f} m^3, LCB {hydro['LCB']:.3f} m from the transom, "
          f"S_wet {hydro['S_wet']:.2f} m^2, LWL {hydro['LWL']:.2f} m, "
          f"variant {'VALID' if hydro['feasible'] == 1.0 else 'NOT VALID'}")
    if args.hydro_curves:
        fr = [float(f) for f in args.drafts.split(",")]
        export_hydro_curves(args.hydro_curves, *hydrostatic_curves(params, fr))

    if not args.no_plot:
        plot_hull_3d(hull_net, show_control_points=True, mirror_port_side=True)
    else:
        print("Batch mode: 3D plot skipped.")
