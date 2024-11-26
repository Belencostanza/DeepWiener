def load_config(file):
    with open(file, 'r') as f:
        line = f.readlines()

    parameters = {}
    for lin in line:
        if "=" in lin:
            # Utiliza '=' como delimitador y elimina posibles espacios
            #try:
            key, value = map(str.strip, lin.strip().split("=", 1))
            parameters[key] = value
            #except ValueError:
            #    print("Error al dividir la línea:", repr(linea))
        #else:
        #    print("Línea sin el formato clave=valor:", repr(linea.strip()))  # strip() aquí

    return parameters


