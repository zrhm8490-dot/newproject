"""
پروژهٔ درس داده‌کاوی: پیش‌بینی و تشخیص دما از روی داده‌های آب‌وهوا

مراحل (طبق فرایند KDD / CRISP-DM):
  1. تولید / بارگذاری داده
  2. شناخت داده (EDA)
  3. پیش‌پردازش: مقادیر گمشده، داده‌های پرت، مهندسی ویژگی
  4. مدل‌سازی: رگرسیون (مقدار دقیق دما) و طبقه‌بندی (سرد / معتدل / گرم)
  5. ارزیابی و مقایسهٔ مدل‌ها

اجرا:  python temperature_prediction.py
خروجی: پوشهٔ output شامل داده، نمودارها و گزارش متنی
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # بدون نیاز به نمایشگر
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# رنگ‌ها: دو رنگ متمایز و سازگار با کوررنگی
BLUE, ORANGE, INK, GRID = "#2563c9", "#d97a1f", "#222222", "#dddddd"
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 11,
        "figure.dpi": 110,
    }
)


# ----------------------------------------------------------------------
# 1) تولید داده (شبیه‌سازی آب‌وهوای یک شهر معتدل)
# ----------------------------------------------------------------------
def make_dataset(n: int = 3000) -> pd.DataFrame:
    day = rng.integers(1, 366, n)
    seasonal = 15 - 12 * np.cos(2 * np.pi * (day - 15) / 365)  # سردترین: دی/ژانویه

    pressure = rng.normal(1013, 8, n)
    wind = np.abs(rng.normal(12, 5, n))
    cloud = np.clip(rng.normal(50, 25, n), 0, 100)

    temp = (
        seasonal
        + 0.10 * (pressure - 1013)  # فشار بالا -> هوای گرم‌تر
        - 0.04 * cloud  # ابری -> کمی خنک‌تر
        - 0.08 * wind  # باد -> خنک‌تر
        + rng.normal(0, 1.8, n)
    )
    humidity = np.clip(65 - 0.8 * (temp - 15) + rng.normal(0, 8, n), 10, 100)

    return pd.DataFrame(
        {
            "day_of_year": day,
            "humidity": humidity.round(1),
            "pressure": pressure.round(1),
            "wind_speed": wind.round(1),
            "cloud_cover": cloud.round(1),
            "temperature": temp.round(2),
        }
    )


def inject_dirty_values(df: pd.DataFrame) -> pd.DataFrame:
    """برای نمایش پیش‌پردازش: مقدار گمشده و داده پرت اضافه می‌کنیم."""
    df = df.copy()
    for col in ["humidity", "pressure", "cloud_cover"]:
        idx = rng.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[idx, col] = np.nan
    idx = rng.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[idx, "wind_speed"] = rng.uniform(90, 160, len(idx)).round(1)  # باد غیرواقعی
    return df


raw = inject_dirty_values(make_dataset())
raw.to_csv(OUT / "weather_raw.csv", index=False)

report = []


def log(text=""):
    print(text)
    report.append(text)


log("=" * 60)
log("1) شناخت داده")
log("=" * 60)
log(f"ابعاد داده: {raw.shape}")
log("\nمقادیر گمشده در هر ستون:")
log(raw.isna().sum().to_string())
log("\nخلاصهٔ آماری (قبل از پاک‌سازی):")
log(raw.describe().round(2).to_string())

# ----------------------------------------------------------------------
# 2) تقسیم داده (قبل از هر چیز، تا نشت اطلاعات نداشته باشیم)
# ----------------------------------------------------------------------
# مهندسی ویژگی: روز سال دایره‌ای است (روز ۳۶۵ کنار روز ۱ است)
raw["doy_sin"] = np.sin(2 * np.pi * raw["day_of_year"] / 365)
raw["doy_cos"] = np.cos(2 * np.pi * raw["day_of_year"] / 365)

FEATURES = ["doy_sin", "doy_cos", "humidity", "pressure", "wind_speed", "cloud_cover"]
X = raw[FEATURES]
y = raw["temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ----------------------------------------------------------------------
# 3) پیش‌پردازش (پارامترها فقط از دادهٔ آموزش یاد گرفته می‌شوند)
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("2) پیش‌پردازش")
log("=" * 60)

# 3-1 داده‌های پرت با روش IQR (برش به کران‌های دادهٔ آموزش)
q1, q3 = X_train["wind_speed"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = int(((X_train["wind_speed"] < low) | (X_train["wind_speed"] > high)).sum())
log(f"داده‌های پرت سرعت باد در آموزش: {n_out} (کران مجاز: {low:.1f} تا {high:.1f})")


def clip_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["wind_speed"] = frame["wind_speed"].clip(low, high)
    return frame


X_train, X_test = clip_outliers(X_train), clip_outliers(X_test)

# 3-2 مقادیر گمشده: جایگزینی با میانه
imputer = SimpleImputer(strategy="median").fit(X_train)
X_train = pd.DataFrame(imputer.transform(X_train), columns=FEATURES, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)
log("مقادیر گمشده با میانهٔ دادهٔ آموزش پر شدند.")

# 3-3 نرمال‌سازی (برای مدل خطی)
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

clean = pd.concat([X_train, X_test]).join(y)
clean.to_csv(OUT / "weather_clean.csv", index=False)

# ----------------------------------------------------------------------
# 4) رگرسیون: پیش‌بینی مقدار دقیق دما
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("3) رگرسیون (پیش‌بینی دمای دقیق، درجهٔ سانتی‌گراد)")
log("=" * 60)

reg_models = {
    "Linear Regression": (LinearRegression(), Xs_train, Xs_test),
    "Random Forest": (
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        X_train.values,
        X_test.values,
    ),
}
reg_results, reg_preds = [], {}
for name, (model, xtr, xte) in reg_models.items():
    model.fit(xtr, y_train)
    pred = model.predict(xte)
    reg_preds[name] = pred
    cv = cross_val_score(model, xtr, y_train, cv=5, scoring="r2").mean()
    reg_results.append(
        {
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
            "CV R2 (5-fold)": cv,
        }
    )
reg_df = pd.DataFrame(reg_results).round(3)
log(reg_df.to_string(index=False))

# ----------------------------------------------------------------------
# 5) طبقه‌بندی: تشخیص «سرد / معتدل / گرم»
# ----------------------------------------------------------------------
log("\n" + "=" * 60)
log("4) طبقه‌بندی (سرد < ۱۰ ≤ معتدل < ۲۰ ≤ گرم)")
log("=" * 60)

BINS, LABELS = [-np.inf, 10, 20, np.inf], ["cold", "mild", "hot"]
yc_train = pd.cut(y_train, BINS, labels=LABELS)
yc_test = pd.cut(y_test, BINS, labels=LABELS)
log("توزیع کلاس‌ها در دادهٔ آموزش:")
log(yc_train.value_counts().reindex(LABELS).to_string())

clf_models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1
    ),
}
clf_results, clf_preds = [], {}
for name, model in clf_models.items():
    model.fit(X_train, yc_train)
    pred = model.predict(X_test)
    clf_preds[name] = pred
    clf_results.append(
        {"Model": name, "Accuracy": accuracy_score(yc_test, pred)}
    )
clf_df = pd.DataFrame(clf_results).round(3)
log("\n" + clf_df.to_string(index=False))

best_clf = max(clf_results, key=lambda r: r["Accuracy"])["Model"]
log(f"\nگزارش کامل بهترین مدل ({best_clf}):")
log(classification_report(yc_test, clf_preds[best_clf], labels=LABELS, digits=3))

# ----------------------------------------------------------------------
# 6) نمودارها
# ----------------------------------------------------------------------
# 6-1 توزیع دما
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(y, bins=40, color=BLUE, edgecolor="white", linewidth=0.8)
for t in (10, 20):
    ax.axvline(t, color=ORANGE, linewidth=1.6, linestyle="--")
ax.set(title="Temperature distribution", xlabel="Temperature (°C)", ylabel="Days")
ax.text(10.3, ax.get_ylim()[1] * 0.92, "cold | mild", color=INK, fontsize=9)
ax.text(20.3, ax.get_ylim()[1] * 0.92, "mild | hot", color=INK, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "01_temperature_distribution.png")
plt.close(fig)

# 6-2 همبستگی ویژگی‌ها
corr = clean[FEATURES + ["temperature"]].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.6 else INK)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Feature correlation")
fig.tight_layout()
fig.savefig(OUT / "02_correlation.png")
plt.close(fig)

# 6-3 مقدار واقعی در برابر پیش‌بینی
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
lims = [y_test.min() - 1, y_test.max() + 1]
for ax, (name, pred), color in zip(axes, reg_preds.items(), (BLUE, ORANGE)):
    ax.scatter(y_test, pred, s=14, alpha=0.5, color=color, edgecolor="none")
    ax.plot(lims, lims, color=INK, linewidth=1)
    r2 = r2_score(y_test, pred)
    ax.set(title=f"{name} (R² = {r2:.3f})", xlabel="Actual (°C)")
axes[0].set_ylabel("Predicted (°C)")
fig.tight_layout()
fig.savefig(OUT / "03_actual_vs_predicted.png")
plt.close(fig)

# 6-4 ماتریس درهم‌ریختگی
fig, ax = plt.subplots(figsize=(5, 4.4))
ConfusionMatrixDisplay(
    confusion_matrix(yc_test, clf_preds[best_clf], labels=LABELS), display_labels=LABELS
).plot(ax=ax, cmap="Blues", colorbar=False)
ax.grid(False)
ax.set_title(f"Confusion matrix - {best_clf}")
fig.tight_layout()
fig.savefig(OUT / "04_confusion_matrix.png")
plt.close(fig)

# 6-5 اهمیت ویژگی‌ها
rf = reg_models["Random Forest"][0]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
ax.set(title="Feature importance (Random Forest regressor)", xlabel="Importance")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "05_feature_importance.png")
plt.close(fig)

log("\n" + "=" * 60)
log("5) نتیجه‌گیری")
log("=" * 60)
top = imp.index[-1]
log(f"- مهم‌ترین ویژگی برای پیش‌بینی دما: {top}")
log(f"- بهترین مدل رگرسیون: {reg_df.sort_values('RMSE').iloc[0]['Model']}")
log(f"- بهترین مدل طبقه‌بندی: {best_clf}")
log(f"فایل‌ها در پوشهٔ {OUT} ذخیره شد.")

(OUT / "report.txt").write_text("\n".join(report), encoding="utf-8")
