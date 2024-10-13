# CIBUR
Cross-view Information Bottleneck solution for Urban region
embedding with Risk awareness(CIBUR)



## Quick Start

### Urban Service

To train and test the CIBUR model for Urban Service tasks across different cities, use the following parameters:

- **CITY_NAME**: `NewYork(NY)`, `Chicago(Chi)`, or `SanFrancisco(SF)`
- **TASK_NAME**: `checkIn`, `crime`, or `serviceCall`

Run the training script as follows:

```bash
python train.py --city CITY_NAME --task TASK_NAME
```

For example, to train the model on the Chicago dataset for crime prediction:

```bash
python train.py --city Chi --task checkIn
```

### Urban Functionality

To perform Urban Functionality analysis using the Manhattan dataset, run the following script:

```bash
python MAN/main.py
```

### Testing with Pre-trained Embeddings

We provide pre-trained embeddings that can be directly used for testing.

- **Urban Service Testing**: Use the following script to test the model with pre-trained embeddings.

  ```bash
  python test_model.py --city CITY_NAME --task TASK_NAME
  ```

  For example, to test the model on the Chicago dataset for the `checkIn` task:

  ```bash
  python test_model.py --city Chi --task checkIn
  ```

- **Urban Functionality Testing (Manhattan)**: To test with the Manhattan dataset, run:

  ```bash
  python MAN/test_result.py
  ```


## Dataset Structure

```
data/
├── data_Chicago/
├── data_NewYork/
├── data_SanFrancisco/
├── tasks_Chicago/
├── tasks_NewYork/
└── tasks_SanFrancisco/
MAN/
└── dataset/
```

## Requirements

- **Python:** 3.11.10
- **Dependencies:** 
  - `torch==2.4.1+cu124`
  - `scikit-learn`
