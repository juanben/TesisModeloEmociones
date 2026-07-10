# ======================================================================
# ARCHIVO: Metricas_Revisores_Fijo.py
# ======================================================================
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from load_windows_loso import load_windows_by_subject, loso_split
from model_lstm_atention import build_lstm_model

WINDOWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Ventanas"))
subjects_data = load_windows_by_subject(WINDOWS_DIR)
all_subjects = list(subjects_data.keys())

print(f"🚀 Evaluando de forma segura los modelos pre-entrenados sobre {len(all_subjects)} sujetos...")

loso_accuracies = []
y_true_all_loso = []
y_pred_all_loso = []

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
        # compile=False evita que Keras intente reconstruir el optimizador dañado de 34 variables
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
    
    print(f"   ➔ Accuracy para {sub_out}: {acc*100:.2f}%")

# ======================================================================
# REPORTES FINALES CON LOS MODELOS COMPLETOS
# ======================================================================
if len(loso_accuracies) > 0:
    mean_loso = np.mean(loso_accuracies)
    std_loso = np.std(loso_accuracies)
    ci_95 = 1.96 * (std_loso / np.sqrt(len(loso_accuracies)))

    print("\n" + "="*60)
    print("📊 MÉTRICAS FINALES RECUPERADAS (MODELOS COMPLETOS)")
    print("="*60)
    print(f"1) Precisión Media LOSO: {mean_loso*100:.2f}% ± {std_loso*100:.2f}%")
    print(f"2) Intervalo de Confianza del 95% (LOSO): [{ (mean_loso - ci_95)*100:.2f}% - {(mean_loso + ci_95)*100:.2f}%]")
    print(f"3) Mejor Sujeto: {all_subjects[np.argmax(loso_accuracies)]} ({np.max(loso_accuracies)*100:.2f}%)")
    print(f"4) Peor Sujeto: {all_subjects[np.argmin(loso_accuracies)]} ({np.min(loso_accuracies)*100:.2f}%)")

    print("\n5) CLASSIFICATION REPORT GLOBAL (LOSO REAL):")
    print(classification_report(y_true_all_loso, y_pred_all_loso, target_names=["Neutral", "Miedo", "Ira", "Alegría"], digits=3))

    print("6) MATRIZ DE CONFUSIÓN CONSOLIDADA LOSO:")
    print(confusion_matrix(y_true_all_loso, y_pred_all_loso))