# X-SEAIPR

This is the X-SEAIPR model. It attempts to model the spread of COVID-19 with time under different proposed interventions.

To clone the repository and enter its working directory:

```bash
>> git clone https://github.com/sanitgupta/X-SEAIPR.git
>> cd X-SEAIPR.git
```

To implement different interventions including ramped up testing, lockdowns and social distancing, make corresponding changes to Model.py

To run the model:

```bash
>> python Simulate.py
```

If you wish to run the model for a particular state, instead of India, replace the files (time series case data, age-wise population data, transportation data) in the Data folder.
