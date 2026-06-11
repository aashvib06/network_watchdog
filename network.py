import os
import h5py
import numpy as np
import pandas as pd

from scipy.signal import windows
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from tqdm import tqdm


FS = 16000
WINDOW_SIZE = 16000
STRIDE = 8000

FUNDAMENTAL = 50

OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# CLEANING
# =====================================================

def clean_signal(x):

    x = np.nan_to_num(x)

    x = x - np.mean(x)

    x = np.clip(x, -10*np.std(x), 10*np.std(x))

    return x

# =====================================================
# ELECTRICAL FEATURES
# =====================================================

def electrical_features(v, i):

    vrms = np.sqrt(np.mean(v**2))

    irms = np.sqrt(np.mean(i**2))

    p = np.mean(v * i)

    s = vrms * irms

    q = np.sqrt(max(s**2 - p**2, 0))

    pf = p / s if s > 0 else 0

    return vrms, irms, p, s, q, pf

# =====================================================
# HARMONICS
# =====================================================

def harmonic_features(i):

    w = windows.hann(len(i))

    spectrum = np.abs(rfft(i * w))

    freqs = rfftfreq(len(i), 1 / FS)

    harmonics = []

    for h in [1,3,5,7,9,11,13]:

        target = FUNDAMENTAL * h

        idx = np.argmin(np.abs(freqs - target))

        harmonics.append(spectrum[idx])

    h1 = harmonics[0]

    if h1 > 0:

        thd = np.sqrt(np.sum(np.square(harmonics[1:]))) / h1

    else:

        thd = 0

    return harmonics, thd, spectrum

# =====================================================
# SPECTRAL FEATURES
# =====================================================

def spectral_features(freqs, spectrum):

    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)

    psd = spectrum**2

    psd = psd / np.sum(psd)

    entropy = -np.sum(psd * np.log(psd + 1e-12))

    return centroid, entropy

# =====================================================
# WAVEFORM FEATURES
# =====================================================

def crest_factor(x):

    return np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-8)

def form_factor(x):

    return np.sqrt(np.mean(x**2)) / (np.mean(np.abs(x)) + 1e-8)

# =====================================================
# VI TRAJECTORY
# =====================================================

def vi_features(v,i):

    area = np.trapz(i, v)

    span = np.max(v) - np.min(v)

    slope = np.mean(np.diff(i)/(np.diff(v)+1e-8))

    return area, span, slope

# =====================================================
# STATISTICS
# =====================================================

def statistical_features(x):

    return [

        np.mean(x),

        np.std(x),

        np.var(x),

        skew(x),

        kurtosis(x)

    ]

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(v,i):

    vrms,irms,p,s,q,pf = electrical_features(v,i)

    harmonics,thd,spectrum = harmonic_features(i)

    freqs = rfftfreq(len(i),1/FS)

    centroid,entropy = spectral_features(freqs,spectrum)

    vcf = crest_factor(v)
    icf = crest_factor(i)

    vff = form_factor(v)
    iff = form_factor(i)

    vi_area,vi_span,vi_slope = vi_features(v,i)

    vstats = statistical_features(v)
    istats = statistical_features(i)

    features = [

        vrms,irms,p,s,q,pf,

        *harmonics,

        thd,

        centroid,
        entropy,

        vcf,
        icf,

        vff,
        iff,

        vi_area,
        vi_span,
        vi_slope,

        *vstats,
        *istats

    ]

    return np.array(features,dtype=np.float32)

# =====================================================
# MAIN LOOP
# =====================================================

def process_house(h5_file, house):

    X_waveform = []
    X_features = []
    Y = []

    with h5py.File(h5_file,"r") as f:

        voltage = f[f"{house}/voltage"][:]

        current = f[f"{house}/current"][:]

        labels = f[f"{house}/labels"][:]

        n = len(voltage)

        for start in tqdm(
            range(0,n-WINDOW_SIZE,STRIDE)
        ):

            end = start + WINDOW_SIZE

            v = voltage[start:end]

            i = current[start:end]

            label = labels[end]

            v = clean_signal(v)

            i = clean_signal(i)

            feats = extract_features(v,i)

            waveform = np.stack([v,i])

            X_waveform.append(waveform)

            X_features.append(feats)

            Y.append(label)

    return (
        np.array(X_waveform),
        np.array(X_features),
        np.array(Y)
    )

# =====================================================
# SAVE
# =====================================================

def save_dataset(Xw,Xf,Y,prefix):

    np.save(
        f"{OUTPUT_DIR}/{prefix}_waveforms.npy",
        Xw
    )

    np.save(
        f"{OUTPUT_DIR}/{prefix}_labels.npy",
        Y
    )

    cols = [

        "Vrms","Irms","P","S","Q","PF",

        "H1","H3","H5","H7","H9","H11","H13",

        "THD",

        "SpectralCentroid",
        "SpectralEntropy",

        "VCrest",
        "ICrest",

        "VForm",
        "IForm",

        "VIArea",
        "VISpan",
        "VISlope",

        "VMean","VStd","VVar","VSkew","VKurt",

        "IMean","IStd","IVar","ISkew","IKurt"
    ]

    df = pd.DataFrame(Xf,columns=cols)

    df["label"] = Y

    df.to_parquet(
        f"{OUTPUT_DIR}/{prefix}_features.parquet"
    )

# =====================================================
# RUN
# =====================================================

houses = ["house_1","house_2","house_5"]

for house in houses:

    Xw,Xf,Y = process_house(
        "ukdale_highfreq.h5",
        house
    )

    save_dataset(Xw,Xf,Y,house)