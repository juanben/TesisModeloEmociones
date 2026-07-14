# ======================================================================
# ARCHIVO: run_pipeline.py
# DESCRIPCIÓN: Ejecutor maestro secuencial del pipeline de emociones
# ======================================================================
import subprocess
import sys
import time

def ejecutar_script(nombre_script):
    """
    Ejecuta un script de Python de forma secuencial y monitorea su estado.
    """
    print("\n" + "="*80)
    print(f"🚀 INICIANDO EJECUCIÓN: {nombre_script}")
    print("="*80)
    
    inicio = time.time()
    
    # Se utiliza sys.executable para garantizar que use el mismo entorno de Python actual (y sus librerías)
    resultado = subprocess.run([sys.executable, nombre_script])
    
    fin = time.time()
    duracion = fin - inicio
    
    if resultado.returncode == 0:
        print(f"\n✔ {nombre_script} finalizado con éxito en {duracion:.2f} segundos.")
    else:
        print(f"\n❌ ERROR CRÍTICO: {nombre_script} falló (código de salida: {resultado.returncode}).")
        print("Pipeline abortado para prevenir inconsistencia en las ventanas de datos.")
        sys.exit(resultado.returncode)

if __name__ == "__main__":
    print("======================================================================")
    # El usuario especificó el orden: Etiquetar, Limpieza y Crear Ventanas
    # Se mapean los nombres físicos de tus archivos corregidos
    pipeline = [
        "etiquetar.py", 
        "LimpiezaArchivos.py", 
        "crear_ventanas.py",
        "verificar_ventanas_full.py"
    ]
    
    print(f"Iniciando pipeline secuencial de 3 fases:")
    for i, script in enumerate(pipeline, 1):
        print(f"  Fase {i}: {script}")
    print("======================================================================")
    
    tiempo_total_inicio = time.time()
    
    # Ejecución ordenada paso a paso
    for script in pipeline:
        ejecutar_script(script)
        
    tiempo_total_fin = time.time()
    duracion_total = tiempo_total_fin - tiempo_total_inicio
    
    print("\n" + "="*80)
    print("🎉 ¡PIPELINE COMPLETADO CON ÉXITO!")
    print(f"Fases procesadas de forma segura. Tiempo total: {duracion_total:.2f} segundos.")
    print("El dataset optimizado ya está empaquetado y listo en la carpeta 'Ventanas'.")
    print("="*80)