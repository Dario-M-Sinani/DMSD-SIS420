import pandas as pd
import re
import uuid
from pathlib import Path

# Configuración de rutas
excel_path = Path('LISTA OFICIAL IBS - SISTEMA ENTREGA NIÑOS.xlsx')
output_sql_path = Path('importacion_final_corregida.sql')

def limpiar_nombre(texto):
    if pd.isna(texto): return ""
    texto = str(texto).strip().upper()
    return re.sub(r'\s+', ' ', texto)

def crear_username(nombre):
    if not nombre: return None
    partes = nombre.lower().split()
    if len(partes) >= 2:
        return (partes[0][0] + partes[1])[:10]
    return partes[0][:10]

def main():
    xls = pd.ExcelFile(excel_path)
    tutores_unicos = {} # {nombre_limpio: {email, password, uuid}}
    estudiantes = []
    vinculos = [] # [(email_tutor, nombre_estudiante, curso)]

    for sheet in xls.sheet_names:
        df = xls.parse(sheet, skiprows=1)
        for _, row in df.iterrows():
            # Extraer datos (ajusta índices si es necesario)
            nombre_est = limpiar_nombre(row.get('Nombre completo del alumno') or row.get('Unnamed: 4'))
            curso = str(row.get('Curso') or row.get('Unnamed: 1'))
            madre = limpiar_nombre(row.get('Nombre completo de la madre/tutora'))
            padre = limpiar_nombre(row.get('Nombre completo del padre/tutor'))

            if not nombre_est or "APELLIDOS" in nombre_est: continue

            # Guardar estudiante
            estudiantes.append((nombre_est, curso))

            # Procesar Tutores evitando duplicados
            for tutor_nombre in [madre, padre]:
                if tutor_nombre and len(tutor_nombre) > 3:
                    if tutor_nombre not in tutores_unicos:
                        user = crear_username(tutor_nombre)
                        # Si el usuario ya existe para OTRO nombre, le ponemos un sufijo
                        original_user = user
                        counter = 1
                        while any(u['email'] == user for u in tutores_unicos.values()):
                            user = f"{original_user}{counter}"
                            counter += 1
                        
                        tutores_unicos[tutor_nombre] = {
                            'nombre': tutor_nombre,
                            'email': user,
                            'qr': str(uuid.uuid4())
                        }
                    
                    vinculos.append((tutores_unicos[tutor_nombre]['email'], nombre_est, curso))

    # GENERAR SQL
    with open(output_sql_path, 'w', encoding='utf-8') as f:
        f.write("BEGIN;\n\n")
        
        # 1. Usuarios (Tutores)
        f.write("-- PASO 1: INSERTAR TUTORES ÚNICOS\n")
        for t in tutores_unicos.values():
            f.write(f"INSERT INTO usuario (nombre_completo, email, password, rol, activo, qr_identifier) \n")
            f.write(f"VALUES ('{t['nombre']}', '{t['email']}', '$2b$12$EjemploHashPassword...', 'padre', true, '{t['qr']}') \n")
            f.write(f"ON CONFLICT (email) DO NOTHING;\n\n")

        # 2. Estudiantes
        f.write("-- PASO 2: INSERTAR ESTUDIANTES\n")
        for est, cur in estudiantes:
            f.write(f"INSERT INTO estudiante (nombre_completo, curso, estado) \n")
            f.write(f"VALUES ('{est}', '{cur}', 'en_aula') \n")
            f.write(f"ON CONFLICT DO NOTHING;\n")

        # 3. Vínculos
        f.write("\n-- PASO 3: VINCULAR PADRES CON HIJOS\n")
        for email, est, cur in vinculos:
            f.write(f"INSERT INTO padreestudiantelink (padre_id, estudiante_id) VALUES (\n")
            f.write(f"  (SELECT id FROM usuario WHERE email = '{email}'),\n")
            f.write(f"  (SELECT id FROM estudiante WHERE nombre_completo = '{est}' AND curso = '{cur}' LIMIT 1)\n")
            f.write(f");\n")

        f.write("\nCOMMIT;")

    print(f"Archivo generado: {output_sql_path}")
    print(f"Tutores únicos detectados: {len(tutores_unicos)}")

if __name__ == '__main__':
    main()
    