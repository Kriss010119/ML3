import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from sklearn import metrics

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="Рекомендации вин", layout="wide")


# 4.5 

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

@st.cache_data
def load_test(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def test_proba(_model, X: pd.DataFrame) -> np.ndarray:
    return _model.predict_proba(X)[:, 1]

@st.cache_data
def curves(y_true: np.ndarray, proba: np.ndarray):
    a, b, _ = metrics.roc_curve(y_true, proba)
    prec, rec, _ = metrics.precision_recall_curve(y_true, proba)
    return (a, b, prec, rec, metrics.roc_auc_score(y_true, proba), metrics.average_precision_score(y_true, proba))



def score_batch(model, batch: pd.DataFrame, features: list, t: float) -> pd.DataFrame:
    if batch.empty:
        raise ValueError("В файле пусто")
    
    missing = [el for el in features if el not in batch.columns]
    if missing:
        raise ValueError("В файле не хватает столбцов: " + ", ".join(missing))

    X = batch[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError("Пропуски в столбцах: " + ", ".join(bad_cols))

    proba = model.predict_proba(X)[:, 1]
    res = batch.copy()
    res["proba_good"] = proba
    res["prediction"] = (proba >= t).astype(int)
    res["decision"] = np.where(res["prediction"] == 1, "рекомендовать", "не рекомендовать")
    return res


def inputs(test: pd.DataFrame, features: list) -> pd.DataFrame:
    values = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        with cols[i % 3]:
            if f == "is_red":
                default = int(test[f].median() >= 0.5)
                kind = st.radio("Тип вина", ["белое", "красное"], index=default,horizontal=True, key="in_is_red")
                values[f] = 1 if kind == "красное" else 0
            else:
                lo, hi, med = float(test[f].min()), float(test[f].max()), float(test[f].median())
                step = (hi - lo) / 200 if hi > lo else 0.01
                values[f] = st.number_input(f, min_value=lo, max_value=hi, value=med,step=step, format="%.4f", key=f"in_{f}")
    return pd.DataFrame([values], columns=features)



st.title("Рекомендации вин: да или нет?")
try:
    model  = load_model(str(APP_DIR / "model.joblib"))
    config = load_config(str(APP_DIR / "config.json"))
    test   = load_test(str(APP_DIR / "test.csv"))

except FileNotFoundError as e:
    st.error(f"Не найден файл {Path(str(e.filename)).name}")
    st.stop()

def_cols = [c for c in test.columns if c != "target"]
FEATURES = list(getattr(model, "feature_names_in_", def_cols))

def_t = float(np.clip(config.get("t", 0.5), 0.0, 1.0))
st.sidebar.header("Настройки")
t = st.sidebar.slider("Порог классификации", 0.0, 1.0, def_t, 0.01, key="t", help="Вино считается хорошим, если вероятность не ниже порога")
st.sidebar.caption(f"Порог: {def_t}")

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(["Оценка вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"])

#4.1

with tab_one:
    st.subheader("Оценка одного вина")
    st.caption("Основную инфу взяли из тестовой выборки.")

    x_one = inputs(test, FEATURES)
    proba_one = float(model.predict_proba(x_one)[0, 1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Вероятность, что вино хорошее", f"{proba_one}")
    c2.metric("Текущий порог", f"{t}")

    if proba_one >= t:
        c3.success("Решение: рекомендовать")
    else:
        c3.warning("Решение: не рекомендовать")

# 4.2

with tab_metrics:
    st.subheader("Качество на тесте")

    if "target" not in test.columns:
        st.error("Нет столбца target")

    else:
        y_true = test["target"].to_numpy()
        proba = test_proba(model, test[FEATURES])
        pred = (proba >= t).astype(int)

        cm = metrics.confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precision = metrics.precision_score(y_true, pred, zero_division=0)
        recall = metrics.recall_score(y_true, pred, zero_division=0)
        f1 = metrics.f1_score(y_true, pred, zero_division=0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{precision}")
        m2.metric("Recall", f"{recall}")
        m3.metric("F1", f"{f1}")

        a, b, prec_c, rec_c, roc_auc, ap = curves(y_true, proba)
        a_pt = fp / (fp + tn) if (fp + tn) else 0.0

        col_cm, col_roc, col_pr = st.columns(3)

        fig_cm = Figure(figsize=(4, 3.8))
        ax = fig_cm.subplots()
        metrics.ConfusionMatrixDisplay(cm, display_labels=["плохое (0)", "хорошее (1)"]).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("Матрица ошибок")
        col_cm.pyplot(fig_cm)

        fig_roc = Figure(figsize=(4, 3.8))
        ax = fig_roc.subplots()
        ax.plot(a, b, label=f"ROC-AUC = {roc_auc}")
        ax.plot([0, 1], [0, 1], "k--", label="случайная модель")
        ax.scatter([a_pt], [recall], color="red", zorder=3, label=f"порог {t}")
        ax.set_xlabel("false positive")
        ax.set_ylabel("true positive")
        ax.set_title("ROC-кривая")
        ax.legend(loc="lower right")
        col_roc.pyplot(fig_roc)

        fig_pr = Figure(figsize=(4, 3.8))
        ax = fig_pr.subplots()
        ax.plot(rec_c, prec_c, label=f"PR-AUC (AP) = {ap}")
        ax.axhline(y_true.mean(), color="k", linestyle="--", label="случайная модель")
        ax.scatter([recall], [precision], color="red", zorder=3, label=f"порог {t}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR-кривая")
        ax.legend(loc="upper right")
        col_pr.pyplot(fig_pr)

# 4.3

with tab_batch:
    st.subheader("Предсказание")
    st.caption(f"Используется порог из сайдбара: {t}. Нужные столбцы: {', '.join(FEATURES)}.")
    uploaded = st.file_uploader("Загрузите CSV с признаками вин (без столбца target)", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            res = score_batch(model, batch, FEATURES, t)
        except pd.errors.EmptyDataError:
            st.error("Файл пустой")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Не удалось прочитать файл")
        else:
            n_good = int(res["prediction"].sum())
            st.success(f"Обработано вин: {len(res)}. Рекомендовано: {n_good}.")
            st.dataframe(res)
            st.download_button("Скачать результат", data=res.to_csv(index=False).encode("utf-8"), file_name="wine_predictions.csv", mime="text/csv")

# 4.4

with tab_coef:
    st.subheader("Коэффициенты логистической регрессии со стандартизированными признаками")
    clf = model.steps[-1][1]
    coef = pd.Series(clf.coef_.ravel(), index=FEATURES).sort_values()

    fig_coef = Figure(figsize=(7, 5))
    ax = fig_coef.subplots()
    ax.barh(coef.index, coef.values, color=["tab:red" if v < 0 else "tab:green" for v in coef])
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("коэффициент маштабированный)")
    fig_coef.tight_layout()
    st.pyplot(fig_coef)

    tp = coef.sort_values(ascending=False).head(3)
    tn = coef.sort_values().head(3)

    st.markdown("Сильнее всего ЗА хорошее вино: " + ", ".join(f"`{k}` ({v})" for k, v in tp.items()))
    st.markdown("Сильнее всего ПРОТИВ: "+ ", ".join(f"`{k}` ({v})" for k, v in tn.items()))
    st.caption("Признаки в вине сильно коррелируют друг с другом, поэтому отдельные коэффициенты стоит интерпретировать осторожно")

