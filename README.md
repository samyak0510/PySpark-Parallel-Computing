# DIC Homework 3: PySpark Parallel Computing

CSE-587 Data Intensive Computing  
University at Buffalo

## Author
**Samyak Bijal Shah**  

---

## Overview

This assignment provides hands-on experience with PySpark for distributed computing. Three tasks are implemented comparing parallel vs sequential approaches.

## Project Structure

```
.
├── src/
│   ├── edit_dist.py       # Edit distance with Spark, multiprocessing, and sequential
│   ├── MLP.py             # MLP classifier inference (Spark vs non-Spark)
│   ├── bird_spark.py      # Bird flock simulation using Spark
│   └── get_gif.py         # Visualization utilities for GIF generation
├── docs/
│   └── samyakbi.pdf       # Technical report with benchmarks and analysis
├── results/
│   └── bird_simulation.gif # Bird flock simulation output
├── .gitignore
└── README.md
```

---

## Task 1.1: Edit Distance Computation

Edit distance measures the minimum number of operations required to transform one string into another. Three implementations are compared:
- **For-loop**: Sequential nested loops
- **Multi-process**: Python multiprocessing
- **Spark**: Distributed computation using PySpark with pandas UDFs

### Usage
```bash
python src/edit_dist.py --csv_dir /path/to/csv --num_sentences n
```

### Results

| Number of Sentences | Spark (s) | Multi-process (s) | For-loop (s) |
|---------------------|-----------|-------------------|--------------|
| 10                  | 15.424    | 1.994             | 0.102        |
| 50                  | 15.991    | 3.748             | 4.280        |
| 100                 | 18.038    | 6.563             | 17.554       |
| 500                 | 21.206    | 104.717           | 405.334      |
| 1000                | 21.874    | –                 | –            |

### Conclusion
The for-loop has scalability issues and quadratic time complexity, making it the slowest for larger datasets. In contrast, multi-process and Spark-based methods leverage parallelism, with Spark being much more efficient at larger scales due to optimized distributed processing.

---

## Task 1.2: MLP Inference

Evaluates inference performance for an MLP classifier using two implementations:
- **Spark-based**: Uses Spark's distributed framework with broadcasted model weights
- **Non-Spark**: Standard PyTorch inference

### Usage
```bash
python src/MLP.py --n_input n --hidden_dim d --hidden_layer l
```

### Results

| n_input | Spark (s) | Non-Spark (s) |
|---------|-----------|---------------|
| 1000    | 28.637    | 0.366         |
| 5000    | 25.108    | 2.064         |
| 10000   | 27.572    | 3.590         |
| 20000   | 27.560    | 6.339         |
| 50000   | 36.807    | 15.405        |
| 100000  | 48.649    | 30.393        |

### Conclusion
Non-Spark MLP outperforms Spark implementation: it executes faster and scales linearly for all input sizes. Spark has higher overhead that restricts its efficiency, making it less suitable for smaller datasets while potentially scalable for larger ones.

---

## Task 1.3: Bird Flock Simulation

A 3D bird flock simulation where each bird's position follows movement rules based on flock dynamics:
- **Alignment**: Birds attempt to stay close to the leader bird
- **Separation**: Birds maintain distance from nearby neighbors
- **Cohesion**: Birds strive to stay with the flock
- **Velocity Constraints**: Flying speed is restricted to a certain range

### Usage
```bash
python src/bird_spark.py --num_birds n
```

### Results

| Number of Birds | Spark (s/frame) | Non-Spark (s/frame) |
|-----------------|-----------------|---------------------|
| 200             | 0.3932          | 0.0178              |
| 1000            | 0.4390          | 0.1354              |
| 5000            | 1.6353          | 1.6229              |

### Conclusion
For smaller flock sizes, the non-Spark implementation outperforms Spark due to its higher overhead. While increasing flock size allows Spark's parallelism to reduce the gap, the non-Spark version maintains a slight performance advantage even at larger scales.

---

## Dependencies

- Python 3.8+
- PySpark
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- RapidFuzz
- tqdm
- OpenCV (cv2)

---

## Output

The bird flock simulation generates:
- Individual frame images in `plot/` directory
- Final animation: `results/bird_simulation.gif`
