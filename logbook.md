#Log Book for project:

##Links:
###Papers
- 1. Levy State Space Model: https://arxiv.org/pdf/1912.12524.pdf
- 2. LSSM for tracking: http://publications.eng.cam.ac.uk/1303074/
- 3. HF futures returns using Langevin: http://publications.eng.cam.ac.uk/455734/
- 4. inverse Gaussian processes: https://arxiv.org/pdf/2105.09429.pdf
- 5. Natural gas modelling: https://www.mdpi.com/1996-1073/12/9/1680/pdf
- 6. Natural gas modelling: https://www.osti.gov/servlets/purl/908487
- 7. More: https://eprints.whiterose.ac.uk/135689/7/Stochastic_Vol_Jumps_Leverage_June_23_2019.pdf

###Data sources
- 1. Possible gas data source: https://tradingeconomics.com/commodity/natural-gas
- 2. Streaming data for Henry Hub Futures: https://uk.investing.com/commodities/natural-gas-streaming-chart
- 3. Buy CME Futures Tick Data
- 4. Gas data up to 1 minute freq: https://www.marketwatch.com/investing/future/ngx21/charts?mod=mw_quote_tab


##14/07/21
- Implemented a gamma process according to paper 1
TO DO
- DONE: ts process
- DONE: reduce repeated code and make more modular
- speed up for loops

##18/10/21
- Implemented the ts process according to paper 1
- Tidied up the code into a more modular form
TO DO
- DONE: check stability of the xi's for both processes, currently liable to overflow
- DONE: get access to the papers
- move on to the GIG process

##Meeting 19/10/21
TO DO
- Normal Gamma process
- beta = gamma^2/2
- need to look for some data - investigate CME gas futures
DONE TODAY:
- Plotting marginal gamma density --> needs a lot of tidying up, density needs to scale with max jump size

## 20/10/21
- CME tick data is only accurate to within a second - there may be several ticks a second
- Data is somewhat volatile but how to deal with the repeated ticks?
- Daily prices do exhibit jumps
- Probably don't need to pay to retrieve enough