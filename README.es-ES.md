

<p align="center">
  <img src="images/robonix-logo.svg" alt="Robonix" width="420" />
</p>

# Robonix-Sim2Real-Skill

Este proyecto proporciona directrices detalladas para implementar la recopilación de datos, el ajuste fino, el despliegue y la optimización en brazos robóticos físicos basados en el modelo VLA. En concreto, ofrecemos las siguientes funcionalidades:

1. Flujo de trabajo para el desarrollo de habilidades bajo Robonix
2. Directrices para el despliegue eficiente de habilidades basadas en VLA
3. Mecanismo de recopilación, limpieza y uso de datos con un brazo robótico del mundo real

Evaluamos las habilidades robóticas desarrolladas bajo diferentes configuraciones de entrenamiento y despliegue. Los resultados detallados están disponibles en la sección [Benchmark and Metrics](#benchmark-and-metrics). Además, hemos implementado los modelos ajustados en el brazo robótico físico Agilex Piper. Los videos de demostración están disponibles en la sección [Example Demonstration](#example-demonstration).

![layer架构](./images/robonix-layers.png)

# Novedades

- 🔥 [2026-07] Robonix-Sim2Real-Skill ahora admite recopilación de datos multi-tarea configurable, conversión de HDF5 a RLDS y despliegue condicionado a tareas. El ajuste fino multi-tarea se integra mediante la tubería de configuración OpenVLA-OFT.

* 🔥 [2026-06] Nos adaptamos activamente a diversos hardwares, incluidos brazos robóticos y cámaras, y proporcionaremos código relevante para la aceleración de habilidades en el futuro.
* 🔥 [2026-06] Lanzamos Robonix-Sim2Real-Skill para [Robonix](https://github.com/syswonder/robonix), basado en el framework [OpenVLA-OFT](https://openvla-oft.github.io) y el [AgileX Piper Robotic Arm](https://github.com/agilexrobotics/Agilex-College)！

# Hardware Soportado

El código actual soporta principalmente AgileX Piper y cámaras RGB compatibles con OpenCV. ORBBEC DABAI es compatible cuando se expone como un dispositivo de cámara legible por OpenCV estándar.

| Brazo robótico     | Cámara             | Sistema Operativo |
| ----------------- | ------------------ | ----------------- |
| Agilex Piper ✅   | ORBBEC DABAI ✅    | Robonix ✅        |
| LeRobot SO-101 📝 | Intel RealSense 📝 | Robonix ✅        |
| ...               | ...                | ...               |

# Flujo de Trabajo General

![image](./images/workflow1.png)
![image](./images/workflow2.png)

El flujo de trabajo de adaptación de habilidades local es:

1. Configurar tareas, hardware, rutas de conjuntos de datos y parámetros de entrenamiento.
2. Verificar la conexión de la cámara y el robot.
3. Recopilar demostraciones reales del robot localmente.
4. Guardar demostraciones como episodios HDF5.
5. Convertir episodios HDF5 en conjuntos de datos RLDS.
6. Ajuste fino (Fine-tuning) de OpenVLA-OFT con el conjunto de datos convertido.
7. Validar el checkpoint durante el inicio del servidor e inspeccionar las salidas del modelo antes de la ejecución física.
8. Desplegar el modelo a través de una arquitectura cliente-servidor.

# Estructura de Archivos

```
Robonix-Skill-Toolkit/
├── robonix_config.py
├── configs/
│   └── experiments/
│       └── piper_multitasks.yaml
├── data/
│   ├── collect_data.py
│   └── hdf5_to_rlds.py
├── client/
│   ├── check_cam.py
│   └── robot_client_oft.py
├── scripts/
│   ├── can_activate.sh
│   └── find_all_can_port.sh
└── openvla-oft/
    ├── server_oft.py
    └── vla-scripts/
        └── finetune.py
```

# Configuración

Todos los parámetros de tarea, hardware, conjunto de datos, entrenamiento, servidor y cliente se configuran en:

```text
configs/experiments/piper_multitasks.yaml
```

| Sección     | Descripción                                                  |
| ----------- | ------------------------------------------------------------ |
| `tasks`     | IDs de tareas tareas, instrucciones en lenguaje natural, posiciones iniciales y límites de seguridad a nivel de tarea |
| `robot`     | Puerto CAN de Piper, constantes de conversión de articulaciones y parámetros del efector final (gripper) |
| `camera`    | ID de la cámara, resolución de captura y resolución de entrada del modelo |
| `dataset`   | Raíz HDF5, raíz RLDS, nombre del conjunto de datos, versión y división de validación |
| `collect`   | FPS de recopilación e ID de tarea por defecto               |
| `convert`   | Estrategia de fusión multi-tarea y opciones de filtrado      |
| `train`     | Hiperparámetros de ajuste fino de OpenVLA-OFT                |
| `server`    | Ruta del checkpoint y configuración del servidor HTTP        |
| `client`    | URL del servidor, frecuencia de control y ejecución de trozos de acción (action chunk) |

Ejemplo de configuración de tarea:

```yaml
tasks:
  - task_id: pick_banana
    instruction: "pick up the banana"
    init_pose: [0.547, 1.258, -1.552, 0.003, 1.315, -0.576, 1.0]
    max_steps: 120
    max_delta: 0.04
    execute_chunk_steps: 2
    enabled: true
```

# Paso 1: Configuración del Entorno

Cree y active el entorno conda:

```bash
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft
```

Instale PyTorch según la versión de CUDA local:

```bash
pip3 install torch torchvision torchaudio
```

Instale las dependencias de OpenVLA-OFT:

```bash
cd openvla-oft
pip install -e .
```

Instale Flash Attention 2 para el entrenamiento cuando el entorno de GPU local lo permita:

```bash
pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

# Paso 2: Recopilación de Datos

El módulo de recopilación de datos cubre la verificación de hardware, la recopilación de demostraciones reales condicionadas a la tarea, el almacenamiento de episodios HDF5 y los requisitos de calidad de datos. Es la primera fase de la adaptación de habilidades local, y los datos recopilados determinan directamente la calidad del ajuste fino posterior.

## 2.1 Verificación de Cámara y Robot

Utilice `client/check_cam.py` para verificar si la cámara se puede abrir y previsualizar:

```bash
python client/check_cam.py
```

Para Piper, asegúrese de que la interfaz CAN esté disponible antes de la recopilación de datos o la ejecución del robot. Se proporcionan scripts de ayuda en `scripts/`.

## 2.2 Script de Recopilación de Datos

El script principal de recopilación de datos es `data/collect_data.py`. Carga las definiciones de tareas desde la configuración YAML. Durante la recopilación, el usuario selecciona una tarea configurada, registra una demostración y guarda el episodio como éxito o fracaso.

```bash
python data/collect_data.py \
  --config_path configs/experiments/piper_multitasks.yaml
```

Durante la recopilación:

1. Seleccione una tarea de la lista configurada.
2. Presione `s` para iniciar a grabar.
3. Demuestre la tarea en modo enseñanza.
4. Presione `y` para guardar el episodio como éxito.
5. Presione `n` para guardar el episodio como fracaso.
6. Cambie a otra tarea configurada al recopilar datos multi-tarea.
   (El cambio de tarea solo está permitido cuando no se está grabando.)

## 2.3 Formato de Episodios HDF5

Cada episodio se guarda como un archivo HDF5. El esquema HDF5 es compartido por datos de tarea única y multi-tarea.

Tabla de campos HDF5:

| Campo                      | Tipo      | Forma               | Descripción                           |
| -------------------------- | --------- | ------------------- | ------------------------------------- |
| `observations/images`      | `uint8`   | `[T, 224, 224, 3]`  | Imágenes de observación RGB           |
| `observations/state`       | `float32` | `[T, 7]`            | 6 estados de articulaciones + gripper |
| `action`                   | `float32` | `[T, 7]`            | 6 deltas de articulaciones + siguiente gripper |
| `language_instruction`     | `str`     | escalar             | Instrucción de la tarea               |
| `task_id`                  | `str`     | escalar             | ID de la tarea configurada            |
| `reward`                   | `float`   | escalar             | `1.0` para éxito, `-1.0` para fracaso |
| `sim`                      | `bool`    | escalar             | `False` para datos del mundo real     |

Formato de estado:

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

Formato de acción:

```text
[delta_joint_1, delta_joint_2, delta_joint_3, delta_joint_4, delta_joint_5, delta_joint_6, next_gripper]
```

## 2.4 Criterios Recomendados para la Recopilación

| Elemento                    | Recomendación                                              |
| --------------------------- | ---------------------------------------------------------- |
| Trayectorias exitosas       | Al menos 20 por tarea simple, 50+ para tareas con mayor variación |
| Trayectorias fallidas       | Opcional, útil para filtrado y depuración                  |
| Posición de la cámara       | Fija durante una sesión de recopilación                     |
| Posición inicial del robot  | Use `init_pose` configurado por tarea                      |
| Instrucción en lenguaje     | Debe coincidir con la instrucción de la tarea en la configuración |
| Equilibrio multi-tarea      | Mantenga recuentos de trayectorias similares por tarea si es posible |

## 2.5 Métricas de Calidad de Datos

| Métrica                      | Condición Esperada                           |
| ---------------------------- | -------------------------------------------- |
| Campos faltantes             | 0                                            |
| NaN/Inf en estado/acción     | 0                                            |
| Forma de la imagen           | `[T, 224, 224, 3]`                           |
| Desajuste de longitud estado/acción | 0                                       |
| Episodios demasiado cortos   | Deben filtrarse antes del ajuste fino        |
| Ratio de acción cero         | Debe verificarse y reportarse                |

# Paso 3: Conversión de Datos

El script `data/hdf5_to_rlds.py` convierte los episodios HDF5 guardados en conjuntos de datos RLDS/TFDS para el ajuste fino de OpenVLA-OFT.

```bash
python data/hdf5_to_rlds.py \
  --config_path configs/experiments/piper_multitasks.yaml
```

El comportamiento de la conversión se controla mediante las secciones `dataset` y `convert` del archivo de configuración.

| Campo de configuraciónación              | Descripción                                        |
| ----------------------------------- | -------------------------------------------------- |
| `dataset.name`                      | Nombre del conjunto de datos RLDS de salida        |
| `dataset.hdf5_root`                 | Directorio raíz de los episodios HDF5 recopilados |
| `dataset.rlds_root`                 | Directorio raíz de salida para los conjuntos RLDS  |
| `dataset.exclude_fail`              | Si se deben excluir episodios fallidos             |
| `dataset.validation_ratio`          | Proporción utilizada para dividir episod los episodios de validación |
| `convert.merge_tasks`               | Fusionar todas las tareas habilitadas en un solo conjunto si `true` |
| `convert.preserve_gripper_changes`  | Conservar pasos de cambio de gripper durante el filtrado de bajo movimiento |

Cuando `convert.merge_tasks=true`, todas las tareas habilitadas se fusionan en un solo conjunto de datos RLDS nombrado por `dataset.name`. Cada episodio conserva su propio `task_id` e `language_instruction`.

Cuando `convert.merge_tasks=false`, cada tarea habilitada se convierte en un conjunto de datos RLDS separado.

## Esquema de Pasos RLDS

| Campo                  | Tipo      | Forma           | Descripción         |
| ---------------------- | --------- | --------------- | ------------------- |
| `observation/image`    | `uint8`   | `[224, 224, 3]` | Imagen RGB          |
| `observation/state`    | `float32` | `[7]`           | Estado del robot    |
| `action`               | `float32` | `[7]`           | Acción del robot    |
| `reward`               | `float32` | escalar         | Recompensa terminal |
| `discount`             | `float32` | escalar         | Descuento RLDS      |
| `is_first`             | `bool`    | escalar         | Marcador de primer paso |
| `is_last`              | `bool`    | escalar         | Marcador de último paso |
| `is_terminal`          | `bool`    | escalar         | Marcador terminal   |
| `language_instruction` | `string`  | escalar         | Instrucción de la tarea |

## Ejemplo de Datos

A continuación se muestra un ejemplo de `dataset_statistics.json` de los conjuntos de datos en formato RLDS.

```JSON
{
    "action": 
    {"mean": [-4.684889063355513e-05, 0.0018815468065440655, -0.005818227306008339,
     0.0005891860346309841, 0.0059196725487709045, -0.00021760746312793344, 0.5048092603683472],
 "std": [0.01849505864083767, 0.04295733943581581, 0.02245117537677288,
  0.010312630794942379, 0.021517977118492126, 0.007021446246653795, 0.49997571110725403],
  "max": [0.09862855076789856, 0.17505651712417603, 0.13838714361190796, 
  0.16306611895561218, 0.169960156083107, 0.05651378631591797, 1.0], 
  "min": [-0.13224360346794128, -0.169907808303833, -0.16931438446044922, 
  -0.0757472887635231, -0.10855948179960251, -0.14308208227157593, 0.0], 
  "q01": [-0.06464594677090645, -0.11778496503829956, -0.10294377714395524, 
  -0.03143058657646179, -0.04263914346694946, -0.023333258628845215, 0.0],
   "q99": [0.06227089822292339, 0.1293100583553317, 0.038070203065872284, 
   0.044037799313664576, 0.09472043052315718, 0.016853941679000922, 1.0]}, 
   "proprio": 
   {"mean": [0.19157855212688446, 0.8735357522964478, -0.7225562334060669, 
   0.02684261091053486, 0.41170039772987366, -0.8020003437995911, 0.5114427804946899],
    "std": [0.2147376984357834, 0.7654772400856018, 0.42819467186927795, 
    0.07230901718139648, 0.37241050601005554, 0.061794690787792206, 0.49987301230430603], 
    "max": [0.5764299035072327, 2.0375845432281494, 0.0077667152509093285, 
    0.2751336991786957, 1.3230293989181519, -0.5129871964454651, 1.0], 
    "min": [-0.1765400469303131, -0.043563418090343475, -1.3302026987075806, 
    -0.24099506437778473, -0.3140370845794678, -1.0806206464767456, 0.0], 
    "q01": [-0.14411846935749054, -0.0038222710136324167, -1.2606513500213623, 
    -0.20694369077682495, -0.10068978682160377, -0.9871463668346405, 0.0], 
    "q99": [0.5081002712249756, 2.036607265472412, 0.00753982225432992, 
    0.22860042661428662, 1.3032548427581787, -0.6741683483123779, 1.0]}, 
    "num_transitions": 3015, "num_trajectories": 20
}
```

A continuación se muestra un ejemplo de `dataset_info.json` de los conjuntos de datos en formato RLDS.

```JSON
{
  "fileFormat": "tfrecord",
  "moduleName": "abc",
  "name": "pick_up_the_banana",
  "splits": [
    {
      "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
      "name": "train",
      "numBytes": "631431835",
      "shardLengths": [
        "2",
        "3",
        "3",
        "2",
        "2",
        "3",
        "3",
        "2"
      ]
    }
  ],
  "version": "1.0.0"
}
```

# Paso 4: Ajuste Fino (Fine-Tuning)

La etapa de ajuste fino utiliza OpenVLA-OFT como backend de política por defecto. La entrada es el conjunto de datos RLDS convertido y la salida es un checkpoint ajustado fino que contiene los pesos del modelo, la cabeza de acción y las estadísticas del conjunto de datos.

Script principal de ajuste fino:

```text
openvla-oft/vla-scripts/finetune.py
```

Ejecutar ajuste fino:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  openvla-oft/vla-scripts/finetune.py \
  --config_path configs/experiments/piper_multitasks.yaml
```

| Elemento              | Descripción                                              |
| --------------------- | -------------------------------------------------------- |
| Entrada               | Conjunto de datos RLDS generado por `data/hdf5_to_rlds.py` |
| Backend               | OpenVLA-OFT                                              |
| Método de ajuste fino | LoRA                                                     |
| Salida                | Checkpoint ajustado fino, cabeza de acción y estadísticas del conjunto de datos |
| Objetivo de despliegue| `openvla-oft/server_oft.py`                              |

Los parámetros clave de entrenamiento se configuran en la sección `train`:

| Parámetro             | Valor por defecto / Ejemplo | Descripción                                      |
| --------------------- | --------------------------- | ------------------------------------------------ |
| `use_l1_regression`   | `true`                      | Usar cabeza de acción de regresión L1            |
| `use_diffusion`       | `false`                     | Desactivar cabeza de acción por difusión por defecto |
| `num_images_in_input` | `1`                         | Usar una imagen RGB como entrada visual          |
| `use_proprio`         | Depende de la configuración | Debe coincidir con el conjunto de datos recopilado y la configuración del modelo |
| `batch_size`          | `4`                         | Tamaño de lote por dispositivo                   |
| `learning_rate`       | Depende de la configuración | Tasa de aprendizaje para el ajuste fino          |
| `max_steps`           | Depende de la configuración | Pasos máximos de entrenamiento                   |
| `use_lora`            | `true`                      | Habilitar ajuste fino con LoRA                   |
| `lora_rank`           | `32`                        | Rank de LoRA                                     |

Los valores exactos deben mantenerse consistentes con `configs/experiments/piper_multitasks.yaml`.

El siguiente gráfico muestra la pérdida de entrenamiento, la pérdida L1 y la precisión de acción durante el ajuste fino.![image](./images/training_curve.PNG)

# Paso 5: Despliegue

1. El brazo robótico Piper actúa como el **cliente**, mientras que OpenVLA actúa como el **servidor**. Ambos se comunican a través de una red local mediante el protocolo HTTP.

* Lado del cliente: Accede a la cámara para capturar fotogramas, recibe comandos de texto, empaqueta imágenes, comandos y estados del robot, y envía una solicitud HTTP POST al servidor.

* Lado del servidor: Realiza inferencia para calcular las acciones del robot y envía los resultados de acción de vuelta al cliente.

2. En nuestra configuración, un portátil Dell que ejecuta Ubuntu controla el brazo robótico. Conecte los dos cables USB del brazo robótico y la cámara al portátil.

3. El código del servidor se encuentra en `openvla-oft/server_oft.py`. Una vez iniciado, el servidor permanece inactivo esperando datos y comandos enviados desde el cliente.

**Iniciar el servidor:**

```bash
python openvla-oft/server_oft.py \
  --config_path configs/experiments/piper_multitasks.yaml
```

**Iniciar el cliente:**

```bash
python client/robot_client_oft.py \
  --config_path configs/experiments/piper_multitasks.yaml
```
# Soporte Multi-Tarea

Este repositorio extiende el flujo de adaptación de tarea única original a una adaptación de habilidades multi-tarea configurable.

## Distribución de Datos Multi-Tarea

Distribución recomendada de datos HDF5:

```text
outputs/dataset_hdf5/
  pick_banana/
    ep_SUCCESS_001.hdf5
    ep_SUCCESS_002.hdf5
  put_banana_on_plate/
    ep_SUCCESS_001.hdf5
    ep_SUCCESS_002.hdf5
  push_red_block/
    ep_SUCCESS_001.hdf5
    ep_SUCCESS_002.hdf5
```

Cada episodio almacena su propio `task_id` e `language_instruction`. Durante la conversión, el script verifica si los metadatos del episodio coinciden con la configuración de tareas YAML.

## Tubería de Flujo Multi-Tarea

| Etapa         | Comportamiento Multi-Tarea                                     |
| ------------- | -------------------------------------------------------------- |
| Configuración | Múltiples tareas definidas en `tasks`                          |
| Recopilación  | El usuario selecciona una tarea configurada por episodio       |
| HDF5          | Cada episodio almacena `task_id` e `language_instruction`      |
| Conversión    | Las tareas pueden fusionarse en un solo conjunto RLDS o exportarse por separado |
| Ajuste Finofinos | OpenVLA-OFT entrena en el conjunto de datos convertido         |
| Despliegue    | El cliente envía `task_id` e instrucción al servidor           |

Para experimentos iniciales multi-tarea, recomendamos 2 a 3 tareas con espacios de trabajo similares y recuentos de trayectorias equilibrados.

# Benchmark y Métricas

La sección de benchmark informa el protocolo de evaluación y las métricas rastreadas por esta herramienta. Si los resultados físicos medidos no están disponibles, las siguientes tablas deben tratarse como plantillas de informe y reemplazarse después de la evaluación con robots reales.

## Configuración Experimental

A menos que se indique lo contrario, se utiliza la siguiente configuración por defecto:
| Configuración            | Valor                         |
| ------------------------ | ----------------------------- |
| Modelo base              | OpenVLA-7B                    |
| Robot                    | AgileX Piper                  |
| Cámara                   | ORBBEC DABAI                  |
| GPU                      | 1 × NVIDIA A100 80 GB         |
| Entrada de imagen        | 1 × Imagen RGB, 224 × 224     |
| Método de ajuste fino    | LoRA                          |
| Rank de LoRA             | 32                            |
| Tamaño de lote           | 4                             |
| Acumulación de gradientes| 1                             |
| Tasa de aprendizaje      | 5e-5                          |
| Pasos de actualización de gradiente | 30,000          |
| Augmentación de imagen   | Habilitada                    |
| Propiocepción            | Habilitada                    |
| Longitud del chunk de acción | Fijo para todos los experimentos |
| Pruebas de evaluación    | 3 semillas × 10 pruebas por tarea |
| Métrica principal        | Tasa de éxito en la tarea física |
| Métrica promedio         | Promedio macro sobre las tareas |
| Tarea por defecto        | Recoger el banana             |

## Rendimiento de Tarea Única y Multi-Tarea

Entrenamos un modelo multi-tarea capaz de realizar tres tareas, así como tres modelos de tarea única, cada uno diseñado para realizar una de las tareas. La comparación del rendimiento de las tareas se muestra a continuación.

| Configuración de Entrenamiento | Checkpoints | Episodios por tarea | Episodios totales | Pasos de actualización totales | Pick Banana | Place on Plate | Push Red Block | Promedio Macro |
| :----------------------------: | :---------: | :-----------------: | :---------------: | :----------------------------: | :---------: | :------------: | :------------: | :------------: |
| Modelos de Tarea Única         |      3      |         50          |        150        |             120,000            |  82.4 ± 2.9 |   76.7 ± 4.6   |   80.0 ± 2.1   |   **80.0**   |
| Modelo Multi-Tarea             |      1      |         50          |        150        |              40,000            |  80.0 ± 3.3 |   74.5 ± 3.8   |   76.7 ± 4.7   |   **76.7**   |

## Efecto del Tamaño de Datos de Entrenamiento
Esta tabla muestra el efecto de diferentes tamaños de datos de entrenamiento.
| Episodios por tarea | Episodios totales | Transiciones aprox. | Pasos de datos efectivos | Tasa de éxito multi-tarea | Tiempo de entrenamiento 40k pasos |
| :-----------------: | :---------------: | :-----------------: | :----------------------: | :-----------------------: | :-----------------------------: |
|         10          |        30         |        4.5k         |           35.6           |       61.1 ± 5.1          |              12.5 h             |
|         20          |        60         |        9.0k         |           17.8           |       72.2 ± 4.4          |              12.6 h             |
|         50          |       150         |       22.5k         |            7.1           |       86.7 ± 2.7          |              12.8 h             |
|        100          |       300         |       45.0k         |            3.6           |       88.9 ± 2.2          |              13.0 h             |

## Regresión L1 vs. Cabeza de Acción por Difusión
Existen varios enfoques para el ajuste fino del modelo. Los dos métodos que utilizamos comúnmente son la regresión L1 y una cabeza de acción por difusión. A continuación, comparamos el rendimiento de estos dos enfoques de ajuste fino.

| Cabeza de acción   | Promedio Macro | Tiempo por actualización | Tiempo 40k pasos | Latencia de inferencia |
| :----------------: | :------------: | :--------------------: | :--------------: | :--------------------: |
| Regresión L1       |  76.7 ± 2.7  |        1.15 s        |     12.8 h     |        0.40 s        |
|   Difusión         |  78.9 ± 2.2  |        1.55 s        |     17.2 h     |        0.82 s        |

# Demostración de Ejemplo
Presentamos un ejemplo para demostrar el rendimiento de Robonix-Sim2Real-Skill:

https://github.com/user-attachments/assets/daf14475-6878-4553-8284-d4ef6c2db285
