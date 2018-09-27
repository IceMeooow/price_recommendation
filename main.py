from price_advisor import Advisor

obj = Advisor('dunnhumby _Breakfast-at-the-Frat/dunnhumby - Breakfast at the Frat.xlsx')
summary_dict = obj.recommend_price()
print(summary_dict)

# product_upc: 1111009477, 2066200530,  4116709428, 7218063979, 88491212971
# store: 15TH & MADISON, VANDALIA, MIDDLETOWN, CROWLEY, HYDE PAR, SOUTHLAKE
