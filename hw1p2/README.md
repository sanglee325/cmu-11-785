# HW1P2: Frame-Level Speech Recognition

## Environment Setting

### Create Conda
* Create conda environment.

    ```bash
    conda create -n flsr python=3.8
    conda activate flsr
    ```

* Install requirements.txt.

    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```

## Run code!

### Train model

* Run `train.py`.

    ```bash
    python train.py
    ```

### Submit result file to kaggle.

* Submit to hw1p2

    ```bash
    kaggle competitions submit -c 11-785-s22-hw1p2 -f <filename.csv> -m "Submmission"
    ```
