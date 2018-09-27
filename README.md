# Price Recommendation 
This project provides an ability to predict the price of product for the next two weeks. Long Short-Term Memory (LSTM) networks were used.

### Data 
***'Breakfast at the Frat'***  data set are used.

### Technology
- python (3.6.2 )
- tensorflow (1.10.0)
- Keras (2.2.2)
- numpy (1.14.5)
- pandas (0.23.4)
- matplotlib (3.0.0)
- seaborn (0.9.0)
- scikit-learn (0.19.2)
- scipy (1.1.0)
- xlrd (1.1.0)

### Project structure
- ***main.py*** - is used to run the project.
- ***demand_forecaster.py*** - allows to predict the demand for a product based on its history.
- ***price_forecaster.py*** - allows to predict the price for a product based on its history.
- ***price_advisor.py*** - allows to recommend product price for text 2 weeks.
- ***data_exploration.ipynb*** - shows data research.
- ***step_by_step_sample.py*** - step by step presentation of the process. 

### How to use
- to recommend product price:
    ```
    python main.py
    ```
>**Note:** User must enter the UPC product code and the name of the store. In addition, the user need to enter promotional information for the next two weeks.

- to see the data research
    ```
    jupyter notebook
    ```
    then open the *data_exploration.ipynb* file.

- to predict product demand and to see model performance:
    ```
    python step_by_step_sample.py
    ```

