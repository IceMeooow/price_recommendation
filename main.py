from forecaster import Forecaster


forecaster = Forecaster('dunnhumby _Breakfast-at-the-Frat/dunnhumby - Breakfast at the Frat.xlsx')
df = forecaster.create_data()

df = forecaster.fill_nan_in_price(df)

price = forecaster.recommend_price(df)
print("Recommended price(s) for next %s week(s) : %s"%(len(price), price.flatten()))
