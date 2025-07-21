import json
import glob
import os

# Buscar todos los archivos con el patrón
json_files = sorted(glob.glob("sim_ck*_t9_vs_ck*_t9.json"))

print(f"🔍 Se encontraron {len(json_files)} archivos JSON.")

# Lista para almacenar resultados
merged_data = []

for filename in json_files:
    print(f"📄 Procesando: {filename}")
    with open(filename, "r") as f:
        try:
            content = json.load(f)
            merged_data.append({
                "file": os.path.basename(filename),
                "data": content
            })
        except json.JSONDecodeError:
            print(f"❌ Error de formato en {filename}. Se omite.")

# Guardar en archivo combinado
output_filename = "merged_similarity_t9.json"
with open(output_filename, "w") as out_file:
    json.dump(merged_data, out_file, indent=2)

print(f"✅ Archivo combinado guardado en '{output_filename}'")
