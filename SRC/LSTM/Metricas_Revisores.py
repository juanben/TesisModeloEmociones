# ======================================================================
# SCRIPT DE SOPORTE PARA REVISORES: generar_metricas_revisores.py
# ======================================================================
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from load_windows_loso import load_windows_by_subject, loso_split
from model_lstm_atention import build_lstm_model  # Ajusta si hay typo en tu nombre de archivo

WINDOWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Ventanas"))
subjects_data = load_windows_by_subject(WINDOWS_DIR)
all_subjects = list(subjects_data.keys())

print(f"🚀 Iniciando evaluación iterativa LOSO sobre {len(all_subjects)} folds (sujetos)...")

loso_accuracies = []
y_true_all_loso = []
y_pred_all_loso = []

# Loop automático sobre todos los sujetos para consolidar la métrica LOSO global
for fold_idx, sub_out in enumerate(all_subjects):
    print(f"\n==================== FOLD {fold_idx+1}/{len(all_subjects)}: Dejando fuera a {sub_out} ====================")
    
    (X_train, y_train, X_val, y_val, 
     X_test_int, y_test_int, X_test_ext, y_test_ext, _, _) = loso_split(subjects_data, sub_out)
    
    # Construir y cargar pesos del modelo entrenado previamente para ese sujeto
    # (Asegúrate de haber corrido run_loso para los sujetos o entrénalo aquí brevemente)
    input_shape = X_train.shape[1:]
    model = build_lstm_model(input_shape=input_shape, num_classes=4)
    
    model_path = f"Modelo/model_attention_LOSO_{sub_out}.keras"
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        print(f"⚠️ No se encontró modelo guardado para {sub_out}. Entrenando por 5 épocas para test...")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64, verbose=0)
        
    # Predicción sobre el sujeto completamente externo (LOSO Puro)
    preds = np.argmax(model.predict(X_test_ext), axis=1)
    
    acc = np.mean(preds == y_test_ext)
    loso_accuracies.append(acc)
    
    y_true_all_loso.extend(y_test_ext)
    y_pred_all_loso.extend(preds)
    
    print(f"➔ Precisión para sujeto {sub_out}: {acc*100:.2f}%")

# ======================================================================
# CÁLCULO DE ESTADÍSTICAS GLOBALES EXIGIDAS POR REVISOR 3
# ======================================================================
mean_loso = np.mean(loso_accuracies)
std_loso = np.std(loso_accuracies)
ci_95 = 1.96 * (std_loso / np.sqrt(len(all_subjects)))

print("\n" + "="*60)
print("📊 REPORTES TÉCNICOS GENERADOS PARA LA CARTA DE RESPUESTA")
print("="*60)
print(f"1) Precisión Media LOSO: {mean_loso*100:.2f}% ± {std_loso*100:.2f}%")
print(f"2) Intervalo de Confianza del 95% (LOSO): [{ (mean_loso - ci_95)*100:.2f}% - {(mean_loso + ci_95)*100:.2f}%]")
print(f"3) Mejor Sujeto: {all_subjects[np.argmax(loso_accuracies)]} ({np.max(loso_accuracies)*100:.2f}%)")
print(f"4) Peor Sujeto: {all_subjects[np.argmin(loso_accuracies)]} ({np.min(loso_accuracies)*100:.2f}%)")

print("\n5) CLASSIFICATION REPORT GLOBAL (LOSO INTER-SUJETO):")
print(classification_report(y_true_all_loso, y_pred_all_loso, target_names=["Neutral", "Miedo", "Ira", "Alegría"], digits=3))

print("6) MATRIZ DE CONFUSIÓN CONSOLIDADA LOSO:")
print(confusion_matrix(y_true_all_loso, y_pred_all_loso))