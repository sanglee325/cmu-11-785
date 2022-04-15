# HW3P2

## Environment Setup

* Create conda environment.

    ```bash
    conda create -n hw3p2 python=3.9
    conda activate hw3p2
    ```

* Install requirements.txt.

    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```

* Install CTCDecode.

    ```bash
    git clone --recursive https://github.com/parlance/ctcdecode.git
    cd ctcdecode
    pip install .
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

* Download Kaggle data from [https://www.kaggle.com/competitions/11-785-s22-hw3p2](https://www.kaggle.com/competitions/11-785-s22-hw3p2).

    ```bash
    mkdir data
    cd data
    kaggle competitions download -c 11-785-s22-hw3p2
    unzip 11-785-s22-hw3p2.zip
    mv hw3p2_student_data/hw3p2_student_data/* ./
    rm -rf hw3p2_student_data
    ```

## Train Model

* Run `train.py`.

    ```bash
    python train.py --name CNN-BILSTM --model biLSTM --epoch 100
    ```
