# HW2P2

## Environment Setup

* Create conda environment.

    ```bash
    conda create -n cmu-idl python=3.9
    conda activate cmu-idl
    ```

* Install requirements.txt.

    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```

## Kaggle Setup

* Install Kaggle and setup kaggle.

    ```bash
    pip install --upgrade --force-reinstall --no-deps kaggle==1.5.8
    mkdir /root/.kaggle

    with open("/root/.kaggle/kaggle.json", "w+") as f:
        f.write('{"username":"","key":""}') # Put your kaggle username & key here

    chmod 600 /root/.kaggle/kaggle.json
    ```

* Download dataset.

    ```bash
    mkdir data
    cd data

    kaggle competitions download -c 11-785-s22-hw2p2-classification
    kaggle competitions download -c 11-785-s22-hw2p2-verification

    unzip -q 11-785-s22-hw2p2-classification.zip
    unzip -q 11-785-s22-hw2p2-verification.zip
    ```
## Run Code

* Train model.

    ```bash
    python train.py \
            --num_workers 8 --epochs 50 --model inceptionv1 \
            --lr 1e-3 --optim adam --loss_type smoothce  \
            --aug_type basic --log_path ./log-cls/ 
    ```

* Verify model.

    ```bash
    python verify.py \
            --model inceptionv1 \
            --model_dir <model_dir path> \
            --model_name <model.pth>
    ```

