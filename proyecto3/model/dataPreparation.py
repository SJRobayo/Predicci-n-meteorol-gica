import pandas as pd

# Lista de rutas de tus archivos CSV. Reemplaza las rutas con las de tu proyecto
csv_paths = [
    'C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/clean_observations.csv',  # Ruta al archivo CSV 1
    'C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/cloudiness.csv',  # Ruta al archivo CSV 2
    'C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/dates.csv',
    'C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/seasons.csv'
    'C:/Users/Sami/PycharmProjects/proyecto 3/proyecto3/csv/weather.csv'
    # Ruta al archivo CSV 3
    # Agrega más rutas según sea necesario
]

# Lista para almacenar cada DataFrame
dataframes = []

# Cargar cada archivo CSV en un DataFrame y agregarlo a la lista
for path in csv_paths:
    try:
        df = pd.read_csv(path)  # Carga el CSV
        dataframes.append(df)  # Agrega el DataFrame a la lista
        print(f"Cargado exitosamente: {path}")
    except Exception as e:
        print(f"Error al cargar el archivo {path}: {e}")

# Combinar todos los DataFrames en uno solo (se realiza una concatenación vertical)
try:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print("DataFrames combinados exitosamente.")
except ValueError as ve:
    print(f"Error al combinar DataFrames: {ve}")

# Ahora puedes trabajar con el DataFrame combinado
# Por ejemplo, muestra las primeras filas:
print(combined_df.head())
