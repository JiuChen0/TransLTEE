# Installation Guide for TransLTEE

This document provides a detailed guide on how to install and run TransLTEE on different operating systems. 

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

## Windows



## MacOS





