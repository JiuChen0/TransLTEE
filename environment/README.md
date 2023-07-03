# Installation Guide for TransLTEE

This document provides a detailed guide on how to install and run TransLTEE on different operating systems. The guide currently covers Ubuntu 22.04, Windows 11, and MacOS. For an easier setup, we've provided Conda environment YAML files for each system. 

For each system, try to import the Conda environment from the corresponding YAML file first. If this process fails, follow the detailed step-by-step guide provided.

## Using Conda environment YAML file

1. Make sure you have installed Anaconda or Miniconda on your system. If not, download it from the official [Anaconda website](https://www.anaconda.com/distribution/).

2. Download the corresponding YAML file for your system.

3. Open a terminal and navigate to the directory where you downloaded the YAML file.

4. Create the new environment from the YAML file:

    ```bash
    conda env create -f TransLTEE.yml
    ```

5. Once the environment is created, activate it:

    ```bash
    conda activate TransLTEE
    ```

6. You should be able to run the main script with:

    ```bash
    python main.py
    ```

If the above process fails for any reason, follow the detailed step-by-step guide below for your system.

## Ubuntu 22.04

1. Make sure you have Python installed. This guide assumes you're using Python 3.11.3. You can verify your Python version by running:

    ```bash
    python --version
    ```

2. Install pip if it's not already installed:

    ```bash
    sudo apt install python3-pip
    ```

3. Create a new Conda environment named "TransLTEE":

    ```bash
    conda create --name TransLTEE python=3.11
    ```

4. Activate the new Conda environment:

    ```bash
    conda activate TransLTEE
    ```

5. Install the following Python libraries in your new environment:

    - TensorFlow:
    
    ```bash
    pip install tensorflow
    ```

    - Transformers:

    ```bash
    pip install transformers
    ```

    - Pandas:

    ```bash
    pip install pandas
    ```

6. Once the required Python libraries are installed, you should be able to run the main script with:

    ```bash
    python main.py
    ```

## Windows 10

1. Make sure you have Python installed. You can download Python from the official website. This guide assumes you're using Python 3.11.3. You can verify your Python version by running:

    ```bash
    python --version
    ```

2. Install pip if it's not already installed. Python installations from python.org include pip by default.

3. Install Anaconda. You can download it from the official Anaconda website.

4. Create a new Conda environment named "TransLTEE":

    ```bash
    conda create --name TransLTEE python=3.11
    ```

5. Activate the new Conda environment:

    ```bash
    conda activate TransLTEE
    ```

6. Install the following Python libraries in your new environment:

    - TensorFlow:
    
    ```bash
    pip install tensorflow
    ```

    - Transformers:

    ```bash
    pip install transformers
    ```

    - Pandas:

    ```bash
    pip install pandas
    ```

7. Once the required Python libraries are installed, you should be able to run the main script with:

    ```bash
    python main.py
    ```

## MacOS

1. Make sure you have Python installed. You can download Python from the official website. This guide assumes you're using Python 3.11.3. You can verify your Python version by running:

    ```bash
    python --version
    ```

2. Install pip if it's not already installed. Python installations from python.org include pip by default.

3. Install Anaconda. You can download it from the official Anaconda website.

4. Create a new Conda environment named "TransLTEE":

    ```bash
    conda create --name TransLTEE python=3.11
    ```

5. Activate the new Conda environment:

    ```bash
    conda activate TransLTEE
    ```

6. Install the following Python libraries in your new environment:

    - TensorFlow:
    
    ```bash
    pip install tensorflow
    ```

    - Transformers:

    ```bash
    pip install transformers
    ```

    - Pandas:

    ```bash
    pip install pandas
    ```

7. Once the required Python libraries are installed, you should be able to run the main script with:

    ```bash
    python main.py
    ```



