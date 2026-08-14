# ======================================================================
# ARCHIVO: Metricas_FoldWise_Completo.py
# Calcula:
#   1) Métricas agregadas (igual que antes) para N=20 y N=19 (sin 1M)
#   2) Métricas FOLD-WISE (media ± SD por clase, IC 95% t-Student)
#      necesarias para reconstruir la Tabla "class-specific performance"
#   3) Matriz de confusión (absoluta y normalizada) con etiquetas EN INGLÉS
# Reutiliza los modelos ya entrenados -- NO vuelve a entrenar nada.
# ======================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from load_windows_loso import load_windows_by_subject, loso_split
from model_lstm_atention import build_lstm_model

WINDOWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Ventanas"))
OUTPUT_DIR = "Resultados_Metricas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDED_SUBJECT = "1M"
CLASS_NAMES_EN = ["Neutral", "Fear", "Anger", "Joy"]
N_CLASSES = 4

subjects_data = load_windows_by_subject(WINDOWS_DIR)
all_subjects = list(subjects_data.keys())

print(f"🚀 Evaluando {len(all_subjects)} sujetos con modelos pre-entrenados...")

# ------------------------------------------------------------
# 1) Evaluar TODOS los folds, guardando métricas POR FOLD y
#    acumulando predicciones para el agregado global.
# ------------------------------------------------------------
per_subject_acc = {}
per_subject_preds = {}
per_subject_true = {}

# Fold-wise per-class metrics: dict[subject] -> array(4,) de precision/recall/f1
foldwise_precision = {}
foldwise_recall = {}
foldwise_f1 = {}
foldwise_support = {}  # para saber qué folds tienen support=0 por clase

for fold_idx, sub_out in enumerate(all_subjects):
    print(f"\n👉 Fold {fold_idx+1}/{len(all_subjects)}: Sujeto de prueba = {sub_out}")

    (X_train, y_train, X_val, y_val,
     X_test_int, y_test_int, X_test_ext, y_test_ext, _, _) = loso_split(subjects_data, sub_out)

    input_shape = X_train.shape[1:]
    model = build_lstm_model(input_shape=input_shape, num_classes=4)

    model_path = f"Modelo/model_attention_LOSO_{sub_out}.keras"
    if not os.path.exists(model_path):
        print(f"   ❌ ERROR: No se encontró el modelo para {sub_out}. Se omite.")
        continue

    trained_model = tf.keras.models.load_model(model_path, compile=False)
    model.set_weights(trained_model.get_weights())

    preds = np.argmax(model.predict(X_test_ext, verbose=0), axis=1)
    acc = np.mean(preds == y_test_ext)

    per_subject_acc[sub_out] = acc
    per_subject_preds[sub_out] = preds
    per_subject_true[sub_out] = y_test_ext

    # ---- Métricas por clase PARA ESTE FOLD ----
    # labels=[0,1,2,3] fuerza a que devuelva las 4 clases aunque
    # alguna tenga support=0 en este fold (precision/recall = 0 en ese caso)
    p, r, f1, sup = precision_recall_fscore_support(
        y_test_ext, preds, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    foldwise_precision[sub_out] = p
    foldwise_recall[sub_out] = r
    foldwise_f1[sub_out] = f1
    foldwise_support[sub_out] = sup

    print(f"   ➔ Accuracy para {sub_out}: {acc*100:.2f}%  |  Support por clase: {sup}")


# ------------------------------------------------------------
# 2) Función: métricas fold-wise (media ± SD, IC t-Student)
#    EXCLUYENDO folds con support=0 en esa clase específica
#    (un support=0 no es "recall 0 real", es "no hubo dato" --
#     incluirlo como 0 distorsiona la media artificialmente)
# ------------------------------------------------------------
def foldwise_summary(subject_list):
    results = {}

    for cls_idx, cls_name in enumerate(CLASS_NAMES_EN):
        precisions, recalls, f1s = [], [], []
        for s in subject_list:
            precisions.append(foldwise_precision[s][cls_idx])
            recalls.append(foldwise_recall[s][cls_idx])
            f1s.append(foldwise_f1[s][cls_idx])

        n = len(precisions)

        def mean_sd_ci(values):
            values = np.array(values)
            mean_v = values.mean()
            sd_v = values.std(ddof=1) if n > 1 else 0.0
            if n > 1:
                t_val = stats.t.ppf(0.975, df=n - 1)
                ci_half = t_val * (sd_v / np.sqrt(n))
            else:
                ci_half = 0.0
            return mean_v, sd_v, (mean_v - ci_half, mean_v + ci_half)

        p_mean, p_sd, p_ci = mean_sd_ci(precisions)
        r_mean, r_sd, r_ci = mean_sd_ci(recalls)
        f_mean, f_sd, f_ci = mean_sd_ci(f1s)

        support_total = sum(foldwise_support[s][cls_idx] for s in subject_list)

        results[cls_name] = {
            "precision": (p_mean, p_sd), "recall": (r_mean, r_sd),
            "f1": (f_mean, f_sd, f_ci), "support": support_total,
            "n_folds_used": n
        }

    accs = [per_subject_acc[s] for s in subject_list]
    n_subj = len(accs)
    acc_mean = np.mean(accs)
    acc_sd = np.std(accs, ddof=1)
    t_val = stats.t.ppf(0.975, df=n_subj - 1)
    acc_ci = (acc_mean - t_val * acc_sd / np.sqrt(n_subj),
              acc_mean + t_val * acc_sd / np.sqrt(n_subj))

    return results, (acc_mean, acc_sd, acc_ci)


def print_and_format_table(subject_list, label):
    results, (acc_mean, acc_sd, acc_ci) = foldwise_summary(subject_list)

    print("\n" + "=" * 70)
    print(f"TABLA FOLD-WISE — {label} (N={len(subject_list)})")
    print("=" * 70)
    header = f"{'Class':<10}{'Precision (M±SD)':<20}{'Recall (M±SD)':<20}{'F1 (M±SD)':<20}{'95% CI [F1]':<20}{'Support':<10}{'Folds used'}"
    print(header)
    for cls_name in CLASS_NAMES_EN:
        r = results[cls_name]
        print(f"{cls_name:<10}"
              f"{r['precision'][0]:.4f}±{r['precision'][1]:.4f}      "
              f"{r['recall'][0]:.4f}±{r['recall'][1]:.4f}      "
              f"{r['f1'][0]:.4f}±{r['f1'][1]:.4f}      "
              f"[{r['f1'][2][0]:.4f}, {r['f1'][2][1]:.4f}]   "
              f"{r['support']:<10}"
              f"{r['n_folds_used']}/{len(subject_list)}")

    macro_p = np.mean([results[c]["precision"][0] for c in CLASS_NAMES_EN])
    macro_p_sd = np.mean([results[c]["precision"][1] for c in CLASS_NAMES_EN])
    macro_r = np.mean([results[c]["recall"][0] for c in CLASS_NAMES_EN])
    macro_r_sd = np.mean([results[c]["recall"][1] for c in CLASS_NAMES_EN])
    macro_f1 = np.mean([results[c]["f1"][0] for c in CLASS_NAMES_EN])
    macro_f1_sd = np.mean([results[c]["f1"][1] for c in CLASS_NAMES_EN])

    total_support = sum(results[c]["support"] for c in CLASS_NAMES_EN)
    weighted_p = sum(results[c]["precision"][0] * results[c]["support"] for c in CLASS_NAMES_EN) / total_support
    weighted_r = sum(results[c]["recall"][0] * results[c]["support"] for c in CLASS_NAMES_EN) / total_support
    weighted_f1 = sum(results[c]["f1"][0] * results[c]["support"] for c in CLASS_NAMES_EN) / total_support

    print(f"\nMacro Average    P={macro_p:.4f}±{macro_p_sd:.4f}  R={macro_r:.4f}±{macro_r_sd:.4f}  F1={macro_f1:.4f}±{macro_f1_sd:.4f}")
    print(f"Weighted Average P={weighted_p:.4f}  R={weighted_r:.4f}  F1={weighted_f1:.4f}")
    print(f"\nBalanced Accuracy (mean of per-class recall): {macro_r*100:.2f}% ± {macro_r_sd*100:.2f}%")
    print(f"Overall Mean Accuracy (LOSO, per-subject): {acc_mean*100:.2f}% ± {acc_sd*100:.2f}% "
          f"(95% CI: [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%])")

    return {
        "per_class": results,
        "macro": (macro_p, macro_p_sd, macro_r, macro_r_sd, macro_f1, macro_f1_sd),
        "weighted": (weighted_p, weighted_r, weighted_f1),
        "balanced_acc": (macro_r, macro_r_sd),
        "overall_acc": (acc_mean, acc_sd, acc_ci),
        "total_support": total_support
    }


full_subjects = list(per_subject_acc.keys())
sensitivity_subjects = [s for s in full_subjects if s != EXCLUDED_SUBJECT]

table_full = print_and_format_table(full_subjects, "COHORTE COMPLETA")
table_sensitivity = print_and_format_table(sensitivity_subjects, f"EXCLUYENDO {EXCLUDED_SUBJECT}")

# ------------------------------------------------------------
# 3) Matriz de confusión (absoluta + normalizada) EN INGLÉS
#    para N=20 (cohorte completa) -- la que va en el paper
# ------------------------------------------------------------
y_true_full = np.concatenate([per_subject_true[s] for s in full_subjects])
y_pred_full = np.concatenate([per_subject_preds[s] for s in full_subjects])
conf_mat_full = confusion_matrix(y_true_full, y_pred_full)
conf_mat_norm = conf_mat_full.astype(float) / conf_mat_full.sum(axis=1, keepdims=True)

print("\n" + "=" * 70)
print("MATRIZ DE CONFUSIÓN ABSOLUTA (N=20) — labels en inglés")
print("=" * 70)
print("Labels:", CLASS_NAMES_EN)
print(conf_mat_full)

print("\nMATRIZ DE CONFUSIÓN NORMALIZADA (N=20)")
print(np.array2string(conf_mat_norm, formatter={'float_kind': lambda x: f"{x:.4f}"}))

# ------------------------------------------------------------
# 4) Gráfica: heatmap con etiquetas en inglés
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.heatmap(conf_mat_full, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES_EN, yticklabels=CLASS_NAMES_EN,
            square=True, cbar=True, annot_kws={"size": 11},
            linewidths=0.5, linecolor='gray')
plt.ylabel('True Affective State', fontsize=11, fontweight='bold')
plt.xlabel('Predicted Affective State', fontsize=11, fontweight='bold')
plt.title('Consolidated LOSO Confusion Matrix (N=20)', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "confusion_matrix_english_N20.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"\n✔ Heatmap (inglés) guardado en: {heatmap_path}")

# ------------------------------------------------------------
# 5) Guardar TODO en un archivo de texto para copiar al paper
# ------------------------------------------------------------
report_path = os.path.join(OUTPUT_DIR, "reporte_foldwise_completo.txt")
with open(report_path, "w", encoding="utf-8") as f:
    for label, subj_list, table in [
        ("FULL COHORT (N=20)", full_subjects, table_full),
        (f"SENSITIVITY ANALYSIS excl. {EXCLUDED_SUBJECT} (N=19)", sensitivity_subjects, table_sensitivity)
    ]:
        f.write("=" * 70 + "\n")
        f.write(f"{label}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Class':<10}{'Precision (M±SD)':<20}{'Recall (M±SD)':<20}{'F1 (M±SD)':<20}{'95% CI [F1]':<22}{'Support':<10}{'Folds used'}\n")
        for cls_name in CLASS_NAMES_EN:
            r = table["per_class"][cls_name]
            f.write(f"{cls_name:<10}"
                    f"{r['precision'][0]:.4f}±{r['precision'][1]:.4f}      "
                    f"{r['recall'][0]:.4f}±{r['recall'][1]:.4f}      "
                    f"{r['f1'][0]:.4f}±{r['f1'][1]:.4f}      "
                    f"[{r['f1'][2][0]:.4f}, {r['f1'][2][1]:.4f}]   "
                    f"{r['support']:<10}"
                    f"{r['n_folds_used']}/{len(subj_list)}\n")

        mp, mpsd, mr, mrsd, mf1, mf1sd = table["macro"]
        wp, wr, wf1 = table["weighted"]
        ba, basd = table["balanced_acc"]
        oa, oasd, oaci = table["overall_acc"]

        f.write(f"\nMacro Average    P={mp:.4f}±{mpsd:.4f}  R={mr:.4f}±{mrsd:.4f}  F1={mf1:.4f}±{mf1sd:.4f}\n")
        f.write(f"Weighted Average P={wp:.4f}  R={wr:.4f}  F1={wf1:.4f}\n")
        f.write(f"Balanced Accuracy: {ba*100:.2f}% ± {basd*100:.2f}%\n")
        f.write(f"Overall Mean Accuracy (LOSO): {oa*100:.2f}% ± {oasd*100:.2f}% "
                f"(95% CI: [{oaci[0]*100:.2f}%, {oaci[1]*100:.2f}%])\n\n")

    f.write("=" * 70 + "\n")
    f.write("CONFUSION MATRIX (Absolute, N=20, English labels)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Labels order: {CLASS_NAMES_EN}\n")
    f.write(np.array2string(conf_mat_full) + "\n\n")
    f.write("CONFUSION MATRIX (Normalized, N=20)\n")
    f.write(np.array2string(conf_mat_norm, formatter={'float_kind': lambda x: f"{x:.4f}"}) + "\n")

print(f"\n✔ Reporte fold-wise completo guardado en: {report_path}")
print("\n🎉 Proceso completo.")