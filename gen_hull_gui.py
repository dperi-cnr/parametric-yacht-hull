"""
Interactive viewer of the hulls of gen_hull.py (PyQt5 + pyvista).

- 3D view of the hull (both sides): surface with adjustable opacity and specular reflection;
  sections, waterlines and buttocks, each selectable, with an adjustable number of curves;
- a column with all the parameters (shape parameters and fixed data): editable value and a slider
  over the bounds (from --limits, a limits.inp / limits_A.inp file, or the documented ranges);
- Save: writes the parameters to params.inp (one row per parameter, "value = name", the format of
  hull_params.dat, readable by gen_hull.py --params);
- Save image: PNG of the 3D view at a selectable resolution scale;
- Exit.
The geometry is computed in a background thread; slider moves are coalesced.

Usage:
    python3 gen_hull_gui.py [--params hull_params.dat] [--limits limits.inp] [--out params.inp]
"""
import os
import sys
import argparse

os.environ.setdefault("QT_API", "pyqt5")

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtCore, QtGui, QtWidgets

import gen_hull as g

# documented bounds of the shape parameters (user manual) and sensible ranges for the fixed data
DEFAULT_BOUNDS = {
    "L": (20.0, 50.0), "beam_rel": (0.0, 1.0), "F_over_L": (0.04, 0.08), "F_bow_over_L": (0.08, 0.14),
    "Cm": (0.70, 0.98), "Cp": (0.58, 0.70), "LCB": (-0.06, 0.005), "bow_fullness": (0.0, 1.0),
    "L_pmb": (0.0, 0.20), "bow_convexity": (0.0, 1.0), "entrance_angle": (0.0, 1.0),
    "transom_imm": (0.0, 0.30), "transom_beam": (0.0, 0.95), "c_transom": (0.55, 0.90),
    "transom_rake": (-10.0, 20.0), "keel_rise_short": (0.0, 1.0), "bilge_softness": (2.0, 8.0),
    "bilge_softness_aft": (2.0, 12.0), "flare_over_L": (0.0, 0.03), "flare_aft_over_L": (-0.01, 0.01),
    "stem_radius_over_L": (0.001, 0.005), "stem_rake": (-30.0, 30.0), "stem_curvature": (-0.25, 0.25),
    "C_BB": (0.08, 0.25), "C_LPR": (0.015, 0.045), "C_ZB": (0.30, 0.62), "bulb_type": (0.0, 1.0),
    "bulb_fair_over_L": (0.05, 0.20), "bulb_rise": (0.0, 0.5), "bulb_implicit": (0.0, 1.0),
    "section_round": (0.0, 1.0), "section_round_aft": (0.0, 1.0), "stern_flare": (0.0, 0.8),
    "stern_lift": (0.0, 1.0), "stern_overhang": (0.0, 0.08), "keel_flat": (0.0, 1.0),
    "bulb_size": (0.0, 1.0), "forefoot": (-1.0, 1.0), "bow_tumble": (0.0, 0.3),
}
SLIDER_STEPS = 1000


def read_bounds(filename):
    """Bounds {name: (lo, hi)} from limits.inp (lo hi # name) or limits_A.inp (base lo hi # name)."""
    out = {}
    with open(filename) as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        if '#' not in line:
            continue
        nums, _, comment = line.partition('#')
        v = nums.split()
        name = comment.strip().partition(' - ')[0].strip()
        if len(v) >= 2 and name:
            out[name] = (float(v[-2]), float(v[-1]))
    return out


def fixed_bounds(p):
    """Ranges for the fixed data around their current values."""
    vol = p.get('volume', 150.0)
    out = {'volume': (0.25 * vol, 2.5 * vol)}
    d = p.get('draft', 0.0)
    out['draft'] = (0.0, max(3.0 * d, 1.0) if d > 0 else 0.5 * p.get('L', 30.0) / 4)
    out['LB_max'] = (0.0, 10.0)
    return out


class HullWorker(QtCore.QObject):
    """Computes the hull surface (and its hydrostatics) for a parameter set, in a background thread."""
    done = QtCore.pyqtSignal(object, object, object)       # params, surface PolyData, info dict
    failed = QtCore.pyqtSignal(object, str)

    @QtCore.pyqtSlot(object, bool)
    def compute(self, p, hydro):
        try:
            p = dict(p)
            fd = g.form_data(p)
            u = np.linspace(0.0, 1.0, 241)
            v = np.linspace(0.0, 1.0, 81)**1.5                   # finer towards the keel
            X, Y, Z = g._hull_surface(p, u[:, None], v[None, :], fd)
            if not np.all(np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)):
                raise ValueError("non-finite surface points")
            Bh, T, H = g.dimensions(p)
            # both sides as one structured surface (port mirrored), then triangulated
            Xb = np.concatenate([X[:, ::-1], X[:, 1:]], axis=1)
            Yb = np.concatenate([-Y[:, ::-1], Y[:, 1:]], axis=1)
            Zb = np.concatenate([Z[:, ::-1], Z[:, 1:]], axis=1)
            surf = pv.StructuredGrid(Xb, Yb, Zb).extract_surface(algorithm='dataset_surface').triangulate()
            # transom face, ruled between the transom line and its centreplane projection (as in the STEP)
            tx, ty, tz = Xb[0], Yb[0], Zb[0]
            face = pv.StructuredGrid(np.stack([tx, tx], 1), np.stack([ty, 0.0 * ty], 1), np.stack([tz, tz], 1))
            surf = surf.merge(face.extract_surface(algorithm='dataset_surface').triangulate()).extract_surface(algorithm='dataset_surface')
            surf = surf.compute_normals(auto_orient_normals=True, split_vertices=False)
            info = {'L': p['L'], 'B': 2.0 * Bh, 'T': T, 'D': H}
            if hydro:
                h = g.compute_hydrostatics(p, 600, 300)
                c = g.feasibility_checks(p, h)
                info.update(disp=h['displacement'], vol=h['volume'], Cb=h['Cb'], LCB=h['LCB'],
                            feasible=c['feasible'], failed=[k for k, val in c.items() if val != 1.0 and k != 'feasible'])
            self.done.emit(p, surf, info)
        except Exception as err:
            self.failed.emit(p, f"{type(err).__name__}: {err}")


class ParamRow(QtWidgets.QWidget):
    """Name, editable value and slider for one parameter."""
    changed = QtCore.pyqtSignal(str, float)

    def __init__(self, name, value, lo, hi, parent=None):
        super().__init__(parent)
        self.name, self.lo, self.hi = name, lo, hi
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lab = QtWidgets.QLabel(name)
        lab.setMinimumWidth(130)
        span = max(abs(hi - lo), 1e-9)
        dec = int(np.clip(3 - np.floor(np.log10(span)), 2, 6))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(dec)
        self.spin.setRange(min(lo, value) - 10 * span, max(hi, value) + 10 * span)
        self.spin.setSingleStep(span / 100.0)
        self.spin.setValue(value)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(95)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setToolTip(f"{lo:g} .. {hi:g}")
        self._set_slider(value)
        lay.addWidget(lab, 0, 0)
        lay.addWidget(self.spin, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)

    def _set_slider(self, value):
        t = 0.0 if self.hi <= self.lo else (value - self.lo) / (self.hi - self.lo)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(np.clip(t, 0.0, 1.0) * SLIDER_STEPS)))
        self.slider.blockSignals(False)

    def _spin_changed(self, value):
        self._set_slider(value)
        self.changed.emit(self.name, float(value))

    def _slider_changed(self, k):
        value = self.lo + (self.hi - self.lo) * k / SLIDER_STEPS
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.changed.emit(self.name, float(self.spin.value()))

    def value(self):
        return float(self.spin.value())


class MainWindow(QtWidgets.QMainWindow):
    request = QtCore.pyqtSignal(object, bool)

    def __init__(self, params, bounds, out_file):
        super().__init__()
        self.setWindowTitle("gen_hull viewer")
        self.params = dict(params)
        self.out_file = out_file
        self.surface = None
        self.actors = {}
        self.pending = None
        self.busy = False
        self.first = True

        # --- 3D view
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        # translucent surface with the curves inside it: depth peeling for correct transparency
        self.plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)

        # --- display controls
        disp = QtWidgets.QGroupBox("Display")
        dl = QtWidgets.QGridLayout(disp)
        self.cb_surf = QtWidgets.QCheckBox("Surface"); self.cb_surf.setChecked(True)
        self.sl_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sl_opacity.setRange(0, 100); self.sl_opacity.setValue(100)
        self.sl_spec = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sl_spec.setRange(0, 100); self.sl_spec.setValue(40)
        dl.addWidget(self.cb_surf, 0, 0)
        dl.addWidget(QtWidgets.QLabel("opacity"), 0, 1); dl.addWidget(self.sl_opacity, 0, 2)
        dl.addWidget(QtWidgets.QLabel("specular"), 1, 1); dl.addWidget(self.sl_spec, 1, 2)
        self.curve_ctl = {}
        for r, (key, label, n0, color) in enumerate((('sections', 'Sections', 21, 'red'), ('waterlines', 'Waterlines', 10, 'blue'),
                                                    ('buttocks', 'Buttocks', 8, 'green')), start=2):
            cb = QtWidgets.QCheckBox(label); cb.setChecked(key == 'sections')
            sp = QtWidgets.QSpinBox(); sp.setRange(1, 80); sp.setValue(n0)
            dl.addWidget(cb, r, 0); dl.addWidget(QtWidgets.QLabel("number"), r, 1); dl.addWidget(sp, r, 2)
            self.curve_ctl[key] = (cb, sp, color)
            cb.toggled.connect(self.redraw); sp.valueChanged.connect(self.redraw)
        self.cb_hydro = QtWidgets.QCheckBox("Hydrostatics and validity (slower)"); self.cb_hydro.setChecked(True)
        dl.addWidget(self.cb_hydro, 5, 0, 1, 3)
        self.cb_surf.toggled.connect(self.redraw)
        self.sl_opacity.valueChanged.connect(self.update_material)
        self.sl_spec.valueChanged.connect(self.update_material)

        # --- stern type: sets the stern parameters to a preset (all of them stay continuous)
        self.cmb_stern = QtWidgets.QComboBox()
        self.cmb_stern.addItems(["(current)"] + list(g.STERN_PRESETS))
        self.cmb_stern.activated[str].connect(self.apply_stern)
        # --- bow type: sets the bow parameters to a preset (all of them stay continuous)
        self.cmb_bow = QtWidgets.QComboBox()
        self.cmb_bow.addItems(["(current)"] + list(g.BOW_PRESETS))
        self.cmb_bow.activated[str].connect(self.apply_bow)
        tl = QtWidgets.QGridLayout()
        tl.addWidget(QtWidgets.QLabel("Stern type"), 0, 0); tl.addWidget(self.cmb_stern, 0, 1)
        tl.addWidget(QtWidgets.QLabel("Bow type"), 1, 0); tl.addWidget(self.cmb_bow, 1, 1)
        tl.setColumnStretch(1, 1)

        # --- parameters
        box = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(box)
        pl.setSpacing(0)
        self.rows = {}
        names = g.PARAM_NAMES + g.FIXED_NAMES + list(g.OPTIONAL_FIXED)
        fb = fixed_bounds(self.params)
        for name in names:
            if name in g.FIXED_NAMES or name in g.OPTIONAL_FIXED:
                if name == g.FIXED_NAMES[0] or name == list(g.OPTIONAL_FIXED)[0]:
                    pl.addWidget(QtWidgets.QLabel("<b>fixed data</b>" if name == g.FIXED_NAMES[0] else
                                                  "<i>draft, LB_max: 0 = not imposed / default</i>"))
            lo, hi = bounds.get(name, DEFAULT_BOUNDS.get(name, fb.get(name, (0.0, 1.0))))
            row = ParamRow(name, float(self.params.get(name, 0.0)), lo, hi)
            row.changed.connect(self.param_changed)
            self.rows[name] = row
            pl.addWidget(row)
        pl.addStretch(1)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(box)
        scroll.setMinimumWidth(330)

        # --- buttons
        bt_import = QtWidgets.QPushButton("Import"); bt_import.clicked.connect(self.import_params)
        bt_import.setToolTip("Load a configuration (hull_params.dat or a saved params.inp)")
        bt_save = QtWidgets.QPushButton("Save"); bt_save.clicked.connect(self.save_params)
        bt_img = QtWidgets.QPushButton("Save image"); bt_img.clicked.connect(self.save_image)
        bt_exit = QtWidgets.QPushButton("Exit"); bt_exit.clicked.connect(self.close)
        bl = QtWidgets.QHBoxLayout(); bl.addWidget(bt_import); bl.addWidget(bt_save); bl.addWidget(bt_img); bl.addWidget(bt_exit)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.addWidget(disp); rl.addLayout(tl); rl.addWidget(scroll, 1); rl.addLayout(bl)
        right.setMaximumWidth(420)
        split = QtWidgets.QSplitter()
        split.addWidget(self.plotter.interactor); split.addWidget(right)
        split.setStretchFactor(0, 4); split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.status = self.statusBar()
        self.resize(1500, 900)

        # --- background computation
        self.thread = QtCore.QThread(self)
        self.worker = HullWorker()
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.compute)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()
        self.timer = QtCore.QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(250)
        self.timer.timeout.connect(self.submit)
        self.submit()

    # ----- computation
    def param_changed(self, name, value):
        self.params[name] = value
        if any(name in v for v in g.STERN_PRESETS.values()):
            self.cmb_stern.setCurrentIndex(0)
        if any(name in v for v in g.BOW_PRESETS.values()):
            self.cmb_bow.setCurrentIndex(0)
        self.timer.start()

    def set_values(self, values):
        """Set several parameters at once (rows and model), then recompute once."""
        for name, value in values.items():
            row = self.rows[name]
            if not row.spin.minimum() <= value <= row.spin.maximum():   # e.g. a ship imported into a yacht session
                span = max(abs(row.hi - row.lo), abs(value), 1e-9)
                row.spin.setRange(min(row.spin.minimum(), value - span), max(row.spin.maximum(), value + span))
            row.spin.blockSignals(True); row.spin.setValue(value); row.spin.blockSignals(False)
            row._set_slider(value)
            self.params[name] = float(value)
        self.timer.start()

    def apply_stern(self, name):
        if name in g.STERN_PRESETS:
            self.set_values(g.STERN_PRESETS[name])
            self.status.showMessage(f"stern type: {name}")

    def apply_bow(self, name):
        if name in g.BOW_PRESETS:
            self.set_values(g.BOW_PRESETS[name])
            self.status.showMessage(f"bow type: {name}")

    def import_params(self, fn=None):
        """Load a configuration file (hull_params.dat format; older versions get the defaults of the
        parameters added later). The slider bounds of the fixed data follow the new values."""
        if not fn:
            fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import configuration", "",
                                                          "Parameter files (*.dat *.inp);;All files (*)")
        if not fn:
            return
        try:
            p = g.load_parameters(fn)
        except Exception as err:
            QtWidgets.QMessageBox.warning(self, "Import", f"{fn}:\n{err}")
            return
        fb = fixed_bounds(p)
        for name in g.FIXED_NAMES + list(g.OPTIONAL_FIXED):
            row = self.rows[name]
            row.lo, row.hi = fb.get(name, (row.lo, row.hi))
        self.set_values({name: float(p[name]) for name in self.rows if name in p})
        self.cmb_stern.setCurrentIndex(0); self.cmb_bow.setCurrentIndex(0)
        self.first = True                     # re-frame the camera on the imported hull
        self.status.showMessage(f"imported {fn}")

    def submit(self):
        if self.busy:
            self.pending = dict(self.params)
            return
        self.busy = True
        self.status.showMessage("computing ...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
        self.request.emit(dict(self.params), self.cb_hydro.isChecked())

    def _finish(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.busy = False
        if self.pending is not None:
            self.params.update(self.pending)
            self.pending = None
            self.submit()

    def on_done(self, p, surf, info):
        self.surface = surf
        msg = f"L {info['L']:.2f} m   B {info['B']:.2f} m   T {info['T']:.3f} m   deck at transom {info['D']:.2f} m"
        if 'disp' in info:
            msg += (f"   displacement {info['disp']:.1f} t   Cb {info['Cb']:.3f}   LCB {info['LCB']:.2f} m   "
                    + ("VALID" if info['feasible'] == 1.0 else "NOT VALID: " + ", ".join(info['failed'])))
        self.status.showMessage(msg)
        self.redraw()
        if self.first:
            self.plotter.view_isometric(); self.plotter.reset_camera(); self.first = False
        self._finish()

    def on_failed(self, p, msg):
        self.status.showMessage("geometry error (previous hull kept): " + msg)
        self._finish()

    # ----- drawing
    def material(self):
        return dict(opacity=self.sl_opacity.value() / 100.0, specular=self.sl_spec.value() / 100.0, specular_power=30.0)

    def update_material(self):
        a = self.actors.get('surface')
        if a is not None:
            m = self.material()
            a.prop.opacity = m['opacity']; a.prop.specular = m['specular']; a.prop.specular_power = m['specular_power']
            self.plotter.render()

    def redraw(self):
        if self.surface is None:
            return
        for a in self.actors.values():
            self.plotter.remove_actor(a, render=False)
        self.actors = {}
        if self.cb_surf.isChecked():
            self.actors['surface'] = self.plotter.add_mesh(self.surface, color="lightsteelblue", smooth_shading=True,
                                                           show_edges=False, **self.material())
        xmin, xmax, ymin, ymax, zmin, zmax = self.surface.bounds
        for key, axis in (('sections', 'x'), ('waterlines', 'z'), ('buttocks', 'y')):
            cb, sp, color = self.curve_ctl[key]
            if not cb.isChecked():
                continue
            n = sp.value()
            if key == 'sections':
                origins = [(x, 0, 0) for x in np.linspace(xmin, xmax, n + 2)[1:-1]]
            elif key == 'waterlines':
                T = g.dimensions(self.params)[1]
                origins = [(0, 0, z) for z in np.linspace(0.0, 1.5 * T, n + 1)[1:]]
            else:
                origins = [(0, y, 0) for y in np.linspace(0.0, ymax, n + 1)[1:]]
                origins += [(0, -o[1], 0) for o in origins]
            lines = [self.surface.slice(normal=axis, origin=o) for o in origins]
            lines = [l for l in lines if l.n_points]
            if lines:
                self.actors[key] = self.plotter.add_mesh(pv.merge(lines), color=color, line_width=2.0)
        self.plotter.render()

    # ----- buttons
    def current_params(self):
        return {name: row.value() for name, row in self.rows.items()}

    def save_params(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save parameters", self.out_file, "Parameter files (*.inp *.dat)")
        if not fn:
            return
        p = self.current_params()
        with open(fn, 'w') as f:
            for name in g.PARAM_NAMES:
                f.write(f"{p[name]:<12.6f} = {name}\n")
            for name in g.FIXED_NAMES + [n for n in g.OPTIONAL_FIXED if p.get(n, 0.0) > 0.0]:
                f.write(f"{p[name]:<12.6f} = {name}\n")
        self.status.showMessage(f"parameters saved to {fn}")

    def save_image(self):
        scale, ok = QtWidgets.QInputDialog.getInt(self, "Save image", "Resolution scale (1 = window size):", 2, 1, 8)
        if not ok:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "hull.png", "PNG images (*.png)")
        if not fn:
            return
        if not fn.lower().endswith('.png'):
            fn += '.png'
        self.plotter.screenshot(fn, scale=scale)
        w, h = self.plotter.window_size
        self.status.showMessage(f"image saved to {fn} ({w * scale} x {h * scale} px)")

    def closeEvent(self, event):
        self.thread.quit(); self.thread.wait(2000)
        self.plotter.close()
        super().closeEvent(event)


def set_gl_format():
    """OpenGL surface with alpha, depth and stencil buffers (needed for the translucent surface)."""
    fmt = QtGui.QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


def main():
    ap = argparse.ArgumentParser(description="Interactive viewer of the gen_hull.py hulls")
    ap.add_argument("--params", default="hull_params.dat", help="initial parameter file")
    ap.add_argument("--limits", default=None, help="bounds for the sliders (limits.inp or limits_A.inp)")
    ap.add_argument("--out", default="params.inp", help="default file name for Save")
    args = ap.parse_args()
    params = g.load_parameters(args.params)
    bounds = read_bounds(args.limits) if args.limits else {}
    set_gl_format()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(params, bounds, args.out)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
