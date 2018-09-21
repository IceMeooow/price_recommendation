# Price Recommendation 
This project provides an ability to predict the price of product for the next week(or weeks). Long Short-Term Memory (LSTM) networks were used.

### Data 
***'Breakfast at the Frat'***  data set are used.

### Technology
- python (3.6.2 )
- tensorflow (1.10.0)
- Keras (2.2.2)
- numpy (1.14.5)
- pandas (0.23.4)
- matplotlib (3.0.0)
- scikit-learn (0.19.2)
- scipy (1.1.0)
- xlrd (1.1.0)

### Project structure
- ***main*** - is used to run the project.
- ***forecaster*** - contains the Forecaster class (all the necessary methods are in this class).
- ***step_by_step_sample*** - step by step presentation of the process. 

### How to use
- to predict product prices
    ```
    python main.py
    ```
>**Note:** product name, store name and period are expected.
- to predict product prices and to see model performance
    ```
    python step_by_step_sample.py
    ```
