# X-SEAIPR: Modelling the spread of COVID-19

This is the X-SEAIPR model. It attempts to model the spread of COVID-19 in India under different proposed interventions.

To clone the repository and enter its working directory:

```bash
>> git clone https://github.com/sanitgupta/X-SEAIPR.git
>> cd X-SEAIPR.git
```

To implement different interventions, including ramped up testing, lockdowns, and social distancing, make corresponding changes to Model.py

To run the model:

```bash
>> python Simulate.py
```

You will now have the results saved in .pkl files.

To generate the plots for each state/district after having run the model:

```bash
>> python PlotNew.py
```

You will now have the plots saved in the Plots folder.

If one wishes to run the model for a particular state (or for India but with newer data), instead of India, one has to replace the data files (time series case data, age-wise population data, transportation data) in the Data folder.

Further, changes can be made to Model.py and Simulate.py to run various other scenarios.

One can edit the params dict in Model.py to change the lockdown dates, to change the dates for which contact reduction is done, to change the amount of contact reduction and to change the date on which testing gets ramped up.

To add more adaptive interventions, for example dynamic lockdowns where a state would shut down once a certain number of cases is reached, one must edit the dx function in Model.py. 

To use one's own estimates of the testing fraction, one can edit the beta.json file in the Data folder.