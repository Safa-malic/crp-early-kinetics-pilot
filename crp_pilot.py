"""
CRP early-kinetics pilot (simulation)
-------------------------------------
Question: can we predict the CRP concentration from only the first
90 seconds of a sensor signal, instead of waiting ~30 minutes?

This is a SIMULATION (fake data made from a formula). It is a proof of
concept, NOT a validation on real sensors.

Run:  python crp_pilot.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # lets the script save plots without opening a window
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)  # fixed seed -> same results every run

# ---------------------------------------------------------------
# 1. SETTINGS (change these and see what happens)
# ---------------------------------------------------------------
N_CURVES   = 1000      # how many fake sensor curves to make
C_MIN, C_MAX = 0.5, 100.0   # concentration range (arbitrary units, log-spaced)
K_ON       = 0.0005    # binding speed (per unit concentration, per second)
K_OFF      = 0.0005    # unbinding speed (per second)
S_MAX      = 100.0     # maximum possible signal
NOISE_SD   = 0.5       # random wiggle size (signal units)
DRIFT_SD   = 0.004     # how much the baseline slides per second (spread)
T_BASE     = 30        # seconds of baseline recorded BEFORE the sample is added
T_EARLY    = 90        # seconds of binding data we are allowed to use
T_END      = 1800      # the "slow" endpoint we want to avoid waiting for (30 min)


# ---------------------------------------------------------------
# 2. MAKE ONE FAKE CURVE
# ---------------------------------------------------------------
def make_curve(conc, k_on_factor=1.0):
    """Return time array and a noisy signal for one sample.

    k_on_factor < 1 mimics a thick (viscous) sample: binding is slower.
    """
    t = np.arange(-T_BASE, T_EARLY + 1)           # time in seconds; 0 = sample added
    k_on = K_ON * k_on_factor * rng.normal(1.0, 0.05)   # small natural variation
    k_obs = k_on * conc + K_OFF                   # observed rate: k_obs = kon*C + koff
    s_eq = S_MAX * conc / (conc + K_OFF / K_ON)   # final level (K_D = koff/kon)

    binding = np.where(t > 0, s_eq * (1 - np.exp(-k_obs * np.clip(t, 0, None))), 0.0)
    drift = rng.normal(0, DRIFT_SD) * t           # slow unwanted slide
    noise = rng.normal(0, NOISE_SD, size=t.size)  # random wiggles
    return t, binding + drift + noise


# ---------------------------------------------------------------
# 3. TURN A CURVE INTO A FEW NUMBERS ("features")
# ---------------------------------------------------------------
def extract_features(t, s):
    base = t < 0
    drift_slope = np.polyfit(t[base], s[base], 1)[0]       # how fast the baseline slides

    early = t >= 0
    # fit a curve  s = a*t^2 + b*t + c  to the first 90 s
    a, b, c = np.polyfit(t[early], s[early], 2)
    initial_slope = b - drift_slope                         # slope with drift removed
    curvature = a
    return [initial_slope, curvature, drift_slope]


# ---------------------------------------------------------------
# 4. BUILD A DATASET
# ---------------------------------------------------------------
def build_dataset(n, k_on_factor_range=(1.0, 1.0)):
    concs = np.exp(rng.uniform(np.log(C_MIN), np.log(C_MAX), n))  # log-spaced mix
    X, y = [], []
    for c in concs:
        factor = rng.uniform(*k_on_factor_range)
        t, s = make_curve(c, factor)
        X.append(extract_features(t, s))
        y.append(np.log10(c))                # predict log10(concentration)
    return np.array(X), np.array(y), concs


def percent_error(y_true_log, y_pred_log):
    """Mean and median % error in the real concentration (not the log)."""
    true_c = 10 ** y_true_log
    pred_c = 10 ** y_pred_log
    err = np.abs(pred_c - true_c) / true_c * 100
    return err.mean(), np.median(err)


# ---------------------------------------------------------------
# 5. TRAIN ON CLEAN DATA, TEST ON CLEAN DATA
# ---------------------------------------------------------------
X, y, concs = build_dataset(N_CURVES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

mean_clean, med_clean = percent_error(y_test, model.predict(X_test))
print(f"Clean conditions:   mean error = {mean_clean:.1f}%   median error = {med_clean:.1f}%")

# ---------------------------------------------------------------
# 6. TEST ON "THICK SAMPLE" DATA (binding 20-40% slower)
#    The model was NOT trained on this, like meeting a new kind of blood.
# ---------------------------------------------------------------
X_visc, y_visc, _ = build_dataset(500, k_on_factor_range=(0.6, 0.8))
mean_visc, med_visc = percent_error(y_visc, model.predict(X_visc))
print(f"Viscous samples:    mean error = {mean_visc:.1f}%   median error = {med_visc:.1f}%")

# ---------------------------------------------------------------
# 7. HOW DOES ERROR CHANGE IF WE WAIT LONGER BEFORE PREDICTING?
#    (This is the "speed vs accuracy" trade-off.)
# ---------------------------------------------------------------
print("\nError vs time window used (clean conditions):")
windows = [30, 60, 90]
errs = []
for w in windows:
    T_EARLY = w                      # change how much data we use
    Xw, yw, _ = build_dataset(N_CURVES)
    Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, random_state=0)
    m = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xtr, ytr)
    mean_w, _ = percent_error(yte, m.predict(Xte))
    errs.append(mean_w)
    print(f"  first {w:>3d} s -> mean error = {mean_w:.1f}%")
T_EARLY = 90                         # put it back

# ---------------------------------------------------------------
# 8. SAVE PICTURES FOR YOUR README
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# left: example curves at 3 concentrations, full 30 min, to show the "bathtub"
tt = np.arange(0, T_END)
for c in [1, 10, 100]:
    k = K_ON * c + K_OFF
    seq = S_MAX * c / (c + K_OFF / K_ON)
    ax[0].plot(tt, seq * (1 - np.exp(-k * tt)), label=f"C = {c}")
ax[0].axvline(90, color="grey", ls="--")
ax[0].text(100, 5, "first 90 s", color="grey")
ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Signal")
ax[0].set_title("Binding curves (noise-free)"); ax[0].legend()

# right: predicted vs true on clean test set
pred = model.predict(X_test)
ax[1].scatter(10 ** y_test, 10 ** pred, s=8, alpha=0.6, label="clean")
pv = model.predict(X_visc)
ax[1].scatter(10 ** y_visc, 10 ** pv, s=8, alpha=0.6, label="viscous")
lims = [C_MIN, C_MAX]
ax[1].plot(lims, lims, "k--", lw=1)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("True concentration"); ax[1].set_ylabel("Predicted (from first 90 s)")
ax[1].set_title("Prediction from early data"); ax[1].legend()

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("\nSaved results.png")
