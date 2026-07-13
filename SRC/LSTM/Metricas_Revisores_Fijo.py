# ======================================================================
# ARCHIVO: Metricas_Revisores_Fijo.py
# ======================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from load_windows_loso import load_windows_by_subject, loso_split
from model_lstm_atention import build_lstm_model

# Configuración de rutas de salida
WINDOWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Ventanas"))
OUTPUT_DIR = "Resultados_Metricas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

subjects_data = load_windows_by_subject(WINDOWS_DIR)
all_subjects = list(subjects_data.keys())

print(f"🚀 Evaluando de forma segura los modelos pre-entrenados sobre {len(all_subjects)} sujetos...")

loso_accuracies = []
y_true_all_loso = []
y_pred_all_loso = []
subject_summary_text = []

for fold_idx, sub_out in enumerate(all_subjects):
    print(f"\n👉 Evaluando Fold {fold_idx+1}/{len(all_subjects)}: Sujeto de prueba = {sub_out}")
    
    # 1) Obtener los datos del split
    (X_train, y_train, X_val, y_val, 
     X_test_int, y_test_int, X_test_ext, y_test_ext, _, _) = loso_split(subjects_data, sub_out)
    
    # 2) Construir la arquitectura limpia desde cero
    input_shape = X_train.shape[1:]
    model = build_lstm_model(input_shape=input_shape, num_classes=4)
    
    # 3) Cargar ÚNICAMENTE los pesos para evitar errores con el optimizador Adam
    model_path = f"Modelo/model_attention_LOSO_{sub_out}.keras"
    
    if os.path.exists(model_path):
        print(f"   ✔ Cargando pesos estructurados desde: {model_path}")
        import tensorflow as tf
        trained_model = tf.keras.models.load_model(model_path, compile=False)
        model.set_weights(trained_model.get_weights())
    else:
        print(f"   ❌ ERROR: No se encontró el modelo para {sub_out} en la ruta {model_path}. Por favor, verifícala.")
        continue
        
    # 4) Realizar predicciones limpias inter-sujeto (LOSO Puro)
    preds = np.argmax(model.predict(X_test_ext), axis=1)
    
    acc = np.mean(preds == y_test_ext)
    loso_accuracies.append(acc)
    
    y_true_all_loso.extend(y_test_ext)
    y_pred_all_loso.extend(preds)
    
    log_line = f"Fold {fold_idx+1}/{len(all_subjects)} - Sujeto: {sub_out} -> Accuracy LOSO: {acc*100:.2f}%"
    subject_summary_text.append(log_line)
    print(f"   ➔ Accuracy para {sub_out}: {acc*100:.2f}%")

# ======================================================================
# REPORTES FINALES CON LOS MODELOS COMPLETOS
# ======================================================================
if len(loso_accuracies) > 0:
    mean_loso = np.mean(loso_accuracies)
    std_loso = np.std(loso_accuracies)
    ci_95 = 1.96 * (std_loso / np.sqrt(len(loso_accuracies)))
    
    best_idx = np.argmax(loso_accuracies)
    worst_idx = np.argmin(loso_accuracies)
    
    class_rep = classification_report(y_true_all_loso, y_pred_all_loso, target_names=["Neutral", "Miedo", "Ira", "Alegría"], digits=3)
    conf_mat = confusion_matrix(y_true_all_loso, y_pred_all_loso)

    # Imprimir en consola de forma estructurada
    print("\n" + "="*60)
    print("📊 MÉTRICAS FINALES RECUPERADAS (MODELOS COMPLETOS)")
    print("="*60)
    print(f"1) Precisión Media LOSO: {mean_loso*100:.2f}% ± {std_loso*100:.2f}%")
    print(f"2) Intervalo de Confianza del 95% (LOSO): [{ (mean_loso - ci_95)*100:.2f}% - {(mean_loso + ci_95)*100:.2f}%]")
    print(f"3) Mejor Sujeto: {all_subjects[best_idx]} ({loso_accuracies[best_idx]*100:.2f}%)")
    print(f"4) Peor Sujeto: {all_subjects[worst_idx]} ({loso_accuracies[worst_idx]*100:.2f}%)")
    print("\n5) CLASSIFICATION REPORT GLOBAL (LOSO REAL):\n", class_rep)
    print("6) MATRIZ DE CONFUSIÓN CONSOLIDADA LOSO:\n", conf_mat)

    # ======================================================================
    # GUARDAR REPORTE DE TEXTO PLANO
    # ======================================================================
    report_path = os.path.join(OUTPUT_DIR, "reporte_final_loso.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("REPORTE OFICIAL DE MÉTRICAS INDEPENDIENTES INTER-SUJETO (LOSO)\n")
        f.write("============================================================\n\n")
        f.write("--- DESGLOSE RENDIMIENTO INDIVIDUAL POR FOLD ---\n")
        for line in subject_summary_text:
            f.write(line + "\n")
        f.write("\n" + "="*60 + "\n")
        f.write("📊 ANÁLISIS ESTADÍSTICO CONSOLIDADO\n")
        f.write("="*60 + "\n")
        f.write(f"Precisión Media LOSO: {mean_loso*100:.2f}% ± {std_loso*100:.2f}%\n")
        f.write(f"Intervalo de Confianza del 95% (LOSO): [{ (mean_loso - ci_95)*100:.2f}% - {(mean_loso + ci_95)*100:.2f}%]\n")
        f.write(f"Mejor Desempeño: Sujeto {all_subjects[best_idx]} ({loso_accuracies[best_idx]*100:.2f}%)\n")
        f.write(f"Peor Desempeño: Sujeto {all_subjects[worst_idx]} ({loso_accuracies[worst_idx]*100:.2f}%)\n\n")
        f.write("--- CLASSIFICATION REPORT GLOBAL ---\n")
        f.write(class_rep + "\n")
        f.write("--- MATRIZ DE CONFUSIÓN CONSOLIDADA ---\n")
        f.write(np.array2string(conf_mat) + "\n")
    
    print(f"\n✔ Reporte escrito de forma exitosa en: {report_path}")

    # ======================================================================
    # 📊 GRAFICA 1: DIAGRAMA DE BARRAS INTER-SUJETO (LOSO)
    # ======================================================================
    plt.figure(figsize=(10, 5.5))
    acc_percentages = [a * 100 for a in loso_accuracies]
    colors = ['#1f77b4' if p >= (mean_loso*100) else '#aec7e8' for p in acc_percentages]
    
    bars = plt.bar(all_subjects, acc_percentages, color=colors, edgecolor='black', alpha=0.85)
    plt.axhline(y=mean_loso*100, color='r', linestyle='--', linewidth=2, 
                label=f'Mean LOSO Accuracy ({mean_loso*100:.2f}%)')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)
        
    plt.ylim(0, 100)
    plt.xlabel('Left-Out Test Participant (Fold)', fontsize=11, fontweight='bold')
    plt.ylabel('External Test Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('Cross-Subject Generalizability: LOSO Performance per Participant', fontsize=12, fontweight='bold', pad=15)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    bar_chart_path = os.path.join(OUTPUT_DIR, "loso_rendimiento_barras.png")
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    print(f"✔ Gráfica de barras inter-sujeto guardada en: {bar_chart_path}")

    # ======================================================================
    # 📊 GRAFICA 2: MAPA DE CALOR DE LA MATRIZ DE CONFUSIÓN CONSOLIDADA
    # ======================================================================
    plt.figure(figsize=(7, 6))
    classes = ['Neutral', 'Miedo', 'Ira', 'Alegría']
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
                square=True, cbar=True, annot_kws={"size": 11}, linewidths=0.5, linecolor='gray')
    
    plt.ylabel('True Affective State', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Affective State', fontsize=11, fontweight='bold')
    plt.title('Consolidated LOSO Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, "loso_matriz_confusion.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"✔ Mapa de calor de la matriz de confusión guardado en: {heatmap_path}")
    print("\n🎉 Todo el procesamiento visual e informativo ha concluido con éxito.")